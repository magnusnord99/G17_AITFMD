"""Build calibrated+clipped+avg3 cubes from raw HistologyHSI-GB data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd 
from typing import Any
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.calibrateClip import calibrate_cube, clip_cube, load_envi_cube
from src.preprocessing.index_dataset import build_dataset_index
from src.preprocessing.spectral_transform import reduce_bands_neighbor_average


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_from_config(config_path: Path, raw_path: str) -> Path:
    return (config_path.parent / raw_path).resolve()


def _save_cube(
    cube: np.ndarray,
    output_root: Path,
    patient_id: str,
    roi_name: str,
    dtype: np.dtype,
) -> Path:
    out_path = output_root / patient_id / f"{roi_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, cube.astype(dtype))
    return out_path


def build_avg3_dataset(
    dataset_root: Path,
    output_root: Path,
    tumor_suffix: str,
    non_tumor_suffix: str,
    eps: float,
    clip_min: float,
    clip_max: float,
    window: int,
    output_dtype: np.dtype = np.float32,
    max_rois: int | None = None,
    verbose: bool = False,
    reduce_bands: bool = True,
    failures: list[dict[str, Any]] | None = None,
):
    """Build avg3 cubes and return saved and failed ROIs."""
    dataset_index = build_dataset_index(
        dataset_root=dataset_root,
        tumor_suffix=tumor_suffix,
        non_tumor_suffix=non_tumor_suffix,
    )
    if failures is None:
        failures = []
    valid_rows = dataset_index[dataset_index["is_valid"]].reset_index(drop=True)

    saved = 0
    iterator = valid_rows.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=len(valid_rows), desc="avg3", unit="roi")

    for _, row in iterator:
        try:
            patient_id = str(row["patient_id"])
            roi_name = str(row["roi_name"])

            raw = load_envi_cube(Path(row["raw_hdr_path"]), Path(row["raw_path"]))
            dark = load_envi_cube(Path(row["dark_hdr_path"]), Path(row["dark_path"]))
            white = load_envi_cube(Path(row["white_hdr_path"]), Path(row["white_path"]))

            calibrated = calibrate_cube(raw, dark, white, eps=eps)
            clipped = clip_cube(calibrated, clip_min=clip_min, clip_max=clip_max)
            if reduce_bands:
                reduced = reduce_bands_neighbor_average(clipped, window=window)
            else:
                reduced = clipped

            out_path = _save_cube(
                reduced,
                output_root,
                patient_id,
                roi_name,
                dtype=output_dtype,
            )
            saved += 1

            if verbose:
                print(
                    f"[{saved}] {patient_id}/{roi_name} "
                    f"raw={raw.shape} reduced={reduced.shape} "
                    f"minmax=({float(reduced.min()):.4f}, {float(reduced.max()):.4f}) "
                    f"dtype={np.dtype(output_dtype).name} -> {out_path}"
                )

            if max_rois is not None and saved >= max_rois:
                break
        except Exception as e:
            failures.append({
                "patient_id": row["patient_id"],
                "roi_name": row["roi_name"],
                "error": str(e),
            })

    return saved, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build avg3 dataset.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/preprocessing.yaml")
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Output dtype for saved .npy cubes.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="avg3",
        help="Subdirectory under calibrated_dir where avg3 cubes are written.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional absolute output root directory. If set, overrides "
            "paths.calibrated_dir from config."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    dataset_root = _resolve_from_config(config_path, cfg["paths"]["input_dataset_root"])
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve() / args.output_subdir
    else:
        output_root = (
            _resolve_from_config(config_path, cfg["paths"]["calibrated_dir"])
            / args.output_subdir
        )
    output_dtype = np.float16 if args.dtype == "float16" else np.float32

    saved, failures = build_avg3_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        tumor_suffix=str(cfg["labels"]["tumor_suffix"]),
        non_tumor_suffix=str(cfg["labels"]["non_tumor_suffix"]),
        eps=float(cfg["calibration"]["eps"]),
        clip_min=float(cfg["calibration"]["clip_min"]),
        clip_max=float(cfg["calibration"]["clip_max"]),
        window=int(cfg["spectral"]["reduction_window"]),
        output_dtype=output_dtype,
        max_rois=args.max_rois,
        verbose=args.verbose,
        reduce_bands=cfg["spectral"]["reduce_bands"],
    )
    print(f"Saved {saved} avg3 cubes to: {output_root} (dtype={np.dtype(output_dtype).name})")
    if failures:
        failures_csv = output_root / "failures.csv"
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
        print(f"Failed ROIs: {len(failures)} (saved to {failures_csv})")


if __name__ == "__main__":
    main()
