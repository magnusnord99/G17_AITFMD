"""Entry point for preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.calibrateClip import calibrate_cube, clip_cube, load_envi_cube
from src.preprocessing.index_dataset import build_dataset_index, save_dataset_index, summarize_index
from src.preprocessing.spectral_transform import reduce_bands_neighbor_average
from src.preprocessing.tissue_mask import build_tissue_mask, save_tissue_mask, tissue_ratio


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_from_config(config_path: Path, raw_path: str) -> Path:
    """Resolve project paths relative to config file location."""
    return (config_path.parent / raw_path).resolve()


def _build_and_save_tissue_masks(
    df: pd.DataFrame,
    masks_dir: Path,
    cfg: dict,
) -> None:
    valid_df = df[df["is_valid"]].copy()
    if valid_df.empty:
        print("[mask] no valid ROIs found; skipping mask generation")
        return

    eps = float(cfg["calibration"]["eps"])
    clip_min = float(cfg["calibration"]["clip_min"])
    clip_max = float(cfg["calibration"]["clip_max"])

    reduce_bands = bool(cfg["spectral"]["reduce_bands"])
    reduction_window = int(cfg["spectral"]["reduction_window"])

    method = str(cfg["tissue_mask"]["method"])
    min_object_size = int(cfg["tissue_mask"]["min_object_size"])
    min_hole_size = int(cfg["tissue_mask"]["min_hole_size"])

    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_paths: list[str] = []
    tissue_ratios: list[float] = []

    print(f"[mask] generating masks for {len(valid_df)} valid ROIs")
    for row in tqdm(valid_df.to_dict(orient="records"), desc="mask", unit="roi"):
        raw = load_envi_cube(Path(row["raw_hdr_path"]), Path(row["raw_path"]))
        dark = load_envi_cube(Path(row["dark_hdr_path"]), Path(row["dark_path"]))
        white = load_envi_cube(Path(row["white_hdr_path"]), Path(row["white_path"]))

        cube = calibrate_cube(raw, dark, white, eps=eps)
        cube = clip_cube(cube, clip_min=clip_min, clip_max=clip_max)

        if reduce_bands:
            cube = reduce_bands_neighbor_average(cube, window=reduction_window)

        mask = build_tissue_mask(
            cube,
            method=method,
            min_object_size=min_object_size,
            min_hole_size=min_hole_size,
        )

        out_path = masks_dir / row["patient_id"] / f"{row['roi_name']}_mask.npy"
        save_tissue_mask(mask, out_path)

        mask_paths.append(str(out_path.resolve()))
        tissue_ratios.append(tissue_ratio(mask))

    valid_df["mask_path"] = mask_paths
    valid_df["tissue_ratio"] = tissue_ratios

    merged = df.merge(
        valid_df[["patient_id", "roi_name", "mask_path", "tissue_ratio"]],
        on=["patient_id", "roi_name"],
        how="left",
    )
    df["mask_path"] = merged["mask_path"]
    df["tissue_ratio"] = merged["tissue_ratio"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing steps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing/preprocessing.yaml",
        help="Path to preprocessing YAML config.",
    )
    parser.add_argument(
        "--build-masks",
        action="store_true",
        help="Generate tissue masks after dataset indexing.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    dataset_root = _resolve_from_config(config_path, cfg["paths"]["input_dataset_root"])
    metadata_csv = _resolve_from_config(config_path, cfg["paths"]["metadata_csv"])
    masks_dir = _resolve_from_config(config_path, cfg["paths"]["masks_dir"])
    tumor_suffix = cfg["labels"]["tumor_suffix"]
    non_tumor_suffix = cfg["labels"]["non_tumor_suffix"]

    print(f"[index] dataset root: {dataset_root}")
    df = build_dataset_index(
        dataset_root=dataset_root,
        tumor_suffix=tumor_suffix,
        non_tumor_suffix=non_tumor_suffix,
    )

    if args.build_masks:
        _build_and_save_tissue_masks(df=df, masks_dir=masks_dir, cfg=cfg)

    out_path = save_dataset_index(df, metadata_csv)
    summary = summarize_index(df)

    print(f"[index] wrote metadata CSV: {out_path}")
    print("[index] summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
