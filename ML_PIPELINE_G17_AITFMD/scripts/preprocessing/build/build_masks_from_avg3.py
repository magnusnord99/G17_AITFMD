"""Build tissue masks from an existing avg3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.tissue_mask import build_tissue_mask, save_tissue_mask, tissue_ratio


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_from_config(config_path: Path, raw_path: str) -> Path:
    return (config_path.parent / raw_path).resolve()


def _iter_avg3_cubes(input_root: Path) -> list[Path]:
    cube_paths: list[Path] = []
    for path in input_root.rglob("*.npy"):
        # Skip macOS metadata files like ._foo.npy
        if path.name.startswith("._"):
            continue
        cube_paths.append(path)
    return sorted(cube_paths)


def build_masks_from_avg3(
    input_root: Path,
    output_root: Path,
    method: str,
    min_object_size: int,
    min_hole_size: int,
    tissue_side: str = "auto",
    target_tissue_ratio: float | None = None,
    max_rois: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    cube_paths = _iter_avg3_cubes(input_root)
    if not cube_paths:
        raise FileNotFoundError(f"No .npy cubes found under: {input_root}")

    rows: list[dict[str, str | float]] = []
    iterator = cube_paths if verbose else tqdm(cube_paths, desc="mask_avg3", unit="roi")

    for cube_path in iterator:
        rel = cube_path.relative_to(input_root)
        if len(rel.parts) < 2:
            # Expecting input_root/patient_id/roi_name.npy
            continue

        patient_id = rel.parts[0]
        roi_name = cube_path.stem

        try:
            cube = np.load(cube_path, allow_pickle=False)
        except (ValueError, OSError) as exc:
            if verbose:
                print(f"Skipping invalid npy file: {cube_path} ({exc})")
            continue
        if cube.ndim != 3:
            raise ValueError(f"Expected cube shape (H, W, C), got {cube.shape} at {cube_path}")

        mask = build_tissue_mask(
            cube,
            method=method,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
            tissue_side=tissue_side,
            target_tissue_ratio=target_tissue_ratio,
        )

        mask_path = output_root / patient_id / f"{roi_name}_mask.npy"
        save_tissue_mask(mask, mask_path)
        ratio = tissue_ratio(mask)

        rows.append(
            {
                "patient_id": str(patient_id),
                "roi_name": str(roi_name),
                "cube_path": str(cube_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "tissue_ratio": float(ratio),
            }
        )

        if max_rois is not None and len(rows) >= max_rois:
            break

        if verbose:
            print(f"{patient_id}/{roi_name}: tissue_ratio={ratio:.4f}")

    manifest_df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mask_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tissue masks from avg3 cubes.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/preprocessing.yaml")
    parser.add_argument(
        "--input-subdir",
        type=str,
        default="avg3",
        help="Subdirectory under paths.calibrated_dir containing avg3 cubes.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="avg3_masks",
        help="Subdirectory under paths.masks_dir where masks are written.",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=None,
        help="Optional absolute avg3 input root. Overrides config paths.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional absolute mask output root. Overrides config paths.",
    )
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument(
        "--tissue-side",
        type=str,
        choices=["auto", "dark", "bright"],
        default=None,
        help="Optional override for tissue side selection in mean_otsu.",
    )
    parser.add_argument(
        "--target-tissue-ratio",
        type=float,
        default=None,
        help="Optional override for auto mode (choose side closest to this ratio).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    if args.input_root:
        input_root = Path(args.input_root).expanduser().resolve()
    else:
        input_root = _resolve_from_config(config_path, cfg["paths"]["calibrated_dir"]) / args.input_subdir

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = _resolve_from_config(config_path, cfg["paths"]["masks_dir"]) / args.output_subdir

    tissue_cfg = cfg["tissue_mask"]
    tissue_side = str(args.tissue_side or tissue_cfg.get("tissue_side", "auto"))
    target_tissue_ratio = args.target_tissue_ratio
    if target_tissue_ratio is None:
        raw_target = tissue_cfg.get("target_tissue_ratio", None)
        target_tissue_ratio = None if raw_target is None else float(raw_target)

    manifest_df = build_masks_from_avg3(
        input_root=input_root,
        output_root=output_root,
        method=str(tissue_cfg["method"]),
        min_object_size=int(tissue_cfg["min_object_size"]),
        min_hole_size=int(tissue_cfg["min_hole_size"]),
        tissue_side=tissue_side,
        target_tissue_ratio=target_tissue_ratio,
        max_rois=args.max_rois,
        verbose=args.verbose,
    )
    print(f"Saved masks: {len(manifest_df)}")
    print(f"Manifest: {output_root / 'mask_manifest.csv'}")


if __name__ == "__main__":
    main()
