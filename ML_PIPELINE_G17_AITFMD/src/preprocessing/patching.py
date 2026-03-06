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
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield patches and top-left (y, x) coordinates."""
    h, w, _ = cube.shape
    if h < patch_h or w < patch_w:
        return

    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            yield cube[y : y + patch_h, x : x + patch_w, :], y, x


def load_numpy_cube(cube_path: Path) -> np.ndarray:
    """Load one .npy cube as float32 with shape (H, W, C)."""
    cube = np.load(cube_path)
    if cube.ndim != 3:
        raise ValueError(f"Cube must have 3 dimensions, got {cube.shape} at {cube_path}")
    return cube.astype(np.float32, copy=False)


def make_patches(
    input_root: Path,
    output_root: Path,
    splits: pd.DataFrame,
    patch_size_h: int,
    patch_size_w: int,
    stride_h: int,
    stride_w: int,
) -> pd.DataFrame:
    """Build patches from all split rows and write a single manifest CSV."""
    manifest_rows: list[dict[str, str | int]] = []

    for _, row in splits.iterrows():
        patient_id = str(row["patient_id"])
        roi_name = str(row["roi_name"])
        split = str(row["split"])
        label_id = int(row["label_id"])

        cube_path = input_root / patient_id / f"{roi_name}.npy"
        if not cube_path.exists():
            continue

        cube = load_numpy_cube(cube_path)
        for patch_idx, (patch, y, x) in enumerate(
            iter_patches(cube, patch_size_h, patch_size_w, stride_h, stride_w)
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
    input_root = Path("data/processed/pca32")
    output_root = Path("data/processed/patches/pca32")
    splits_path = Path("data/splits/patient_split.csv")
    splits = get_patient_splits(splits_path)
    manifest_df = make_patches(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        patch_size_h=64,
        patch_size_w=64,
        stride_h=32,
        stride_w=32,
    )
    print(f"saved patches: {len(manifest_df)}")
    print(f"manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()


