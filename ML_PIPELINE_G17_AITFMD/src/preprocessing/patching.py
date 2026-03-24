from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def get_patient_splits(csv_path: Path) -> pd.DataFrame:
    """Load split CSV and validate minimum required columns."""
    df = pd.read_csv(csv_path)
    required = {"patient_id", "roi_name", "split", "label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def iter_patches(
    cube: np.ndarray,
    patch_h: int,
    patch_w: int,
    stride_h: int,
    stride_w: int,
    mask: np.ndarray | None = None,
    min_tissue_ratio: float = 0.0,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield patches and top-left (y, x) coordinates.

    If mask is provided and min_tissue_ratio > 0, only yields patches where
    the fraction of tissue pixels (mask > 0) is at least min_tissue_ratio.
    """
    h, w, _ = cube.shape
    if h < patch_h or w < patch_w:
        return

    use_mask = mask is not None and min_tissue_ratio > 0
    if use_mask and mask.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {mask.shape[:2]} must match cube spatial {h}x{w}")

    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            if use_mask:
                patch_mask = mask[y : y + patch_h, x : x + patch_w]
                tissue_frac = float(np.mean(patch_mask > 0))
                if tissue_frac < min_tissue_ratio:
                    continue
            yield cube[y : y + patch_h, x : x + patch_w, :], y, x


def count_patch_grid(
    cube: np.ndarray,
    patch_h: int,
    patch_w: int,
    stride_h: int,
    stride_w: int,
    mask: np.ndarray | None = None,
    min_tissue_ratio: float = 0.0,
) -> dict[str, int]:
    """
    Tell grid-posisjoner med samme logikk som iter_patches (uten å kutte ut data).

    Returns:
        total_possible: alle (y,x) som passer i kuben med gitt stride
        filtered_by_tissue: hoppet pga. for lav vev-andel i patch-vindu
        evaluated: total_possible - filtered_by_tissue (matcher antall yields fra iter_patches)
    """
    h, w, _ = cube.shape
    if h < patch_h or w < patch_w:
        return {"total_possible": 0, "filtered_by_tissue": 0, "evaluated": 0}

    use_mask = mask is not None and min_tissue_ratio > 0
    if use_mask and mask.shape[:2] != (h, w):
        raise ValueError(f"Mask shape {mask.shape[:2]} must match cube spatial {h}x{w}")

    total = 0
    filtered = 0
    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            total += 1
            if use_mask:
                patch_mask = mask[y : y + patch_h, x : x + patch_w]
                tissue_frac = float(np.mean(patch_mask > 0))
                if tissue_frac < min_tissue_ratio:
                    filtered += 1

    return {
        "total_possible": total,
        "filtered_by_tissue": filtered,
        "evaluated": total - filtered,
    }


def load_numpy_cube(cube_path: Path) -> np.ndarray:
    """Load one .npy cube as float32 with shape (H, W, C)."""
    cube = np.load(cube_path)
    if cube.ndim != 3:
        raise ValueError(f"Cube must have 3 dimensions, got {cube.shape} at {cube_path}")
    return cube.astype(np.float32, copy=False)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load tissue mask as (H, W) with values 0 (background) or 1 (tissue)."""
    mask = np.load(mask_path, allow_pickle=False)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape} at {mask_path}")
    return mask


def make_patches(
    input_root: Path,
    output_root: Path,
    splits: pd.DataFrame,
    patch_size_h: int,
    patch_size_w: int,
    stride_h: int,
    stride_w: int,
    mask_root: Path | None = None,
    min_tissue_ratio: float = 0.0,
) -> pd.DataFrame:
    """Build patches from all split rows and write a single manifest CSV.

    If mask_root is set and min_tissue_ratio > 0, only patches with at least
    that fraction of tissue pixels (from mask) are kept. Masks are expected at
    mask_root/patient_id/{roi_name}_mask.npy.
    """
    manifest_rows: list[dict[str, str | int]] = []
    use_mask = mask_root is not None and min_tissue_ratio > 0

    for _, row in splits.iterrows():
        patient_id = str(row["patient_id"])
        roi_name = str(row["roi_name"])
        split = str(row["split"])
        label_id = int(row["label_id"])

        cube_path = input_root / patient_id / f"{roi_name}.npy"
        if not cube_path.exists():
            continue

        cube = load_numpy_cube(cube_path)
        mask = None
        if use_mask:
            mask_path = mask_root / patient_id / f"{roi_name}_mask.npy"
            if mask_path.exists():
                mask = load_mask(mask_path)
            # else: no mask for this ROI, include all patches

        for patch_idx, (patch, y, x) in enumerate(
            iter_patches(
                cube,
                patch_size_h,
                patch_size_w,
                stride_h,
                stride_w,
                mask=mask,
                min_tissue_ratio=min_tissue_ratio,
            )
        ):
            patch_id = f"{patient_id}_{roi_name}_{y}_{x}_{patch_idx}"
            patch_path = output_root / split / str(label_id) / f"{patch_id}.npy"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(patch_path, patch.astype(np.float32, copy=False))

            manifest_rows.append(
                {
                    "patch_id": patch_id,
                    "patch_path": str(patch_path),
                    "patient_id": patient_id,
                    "roi_name": roi_name,
                    "split": split,
                    "label_id": label_id,
                    "y": y,
                    "x": x,
                    "h": patch.shape[0],
                    "w": patch.shape[1],
                    "c": patch.shape[2],
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def main() -> None:
    """Standalone entry with hardcoded defaults. Prefer build_patches.py for config-driven runs."""
    input_root = Path("data/processed/pca32")
    output_root = Path("data/processed/patches/pca32")
    splits_path = Path("data/splits/patient_split.csv")
    mask_root = Path("data/interim/masks/final_mask")
    min_tissue_ratio = 0.60

    splits = get_patient_splits(splits_path)
    effective_mask_root = mask_root if mask_root.exists() else None
    if effective_mask_root:
        print(f"Using tissue masks from {effective_mask_root} (min_tissue_ratio={min_tissue_ratio})")
    else:
        print(f"Mask root {mask_root} not found; including all patches (no mask filter)")

    manifest_df = make_patches(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        patch_size_h=64,
        patch_size_w=64,
        stride_h=32,
        stride_w=32,
        mask_root=effective_mask_root,
        min_tissue_ratio=min_tissue_ratio,
    )
    print(f"saved patches: {len(manifest_df)}")
    print(f"manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()


