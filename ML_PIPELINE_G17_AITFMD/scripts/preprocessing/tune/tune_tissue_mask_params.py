"""Grid-search tissue mask morphology parameters on avg3 cubes."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing.build.build_masks_from_avg3 import build_masks_from_avg3, _load_yaml


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _label_from_roi(roi_name: str) -> str:
    if roi_name.endswith("_T"):
        return "T"
    if roi_name.endswith("_NT"):
        return "NT"
    return "UNK"


def _ratio_stats(df: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    ratios = df["tissue_ratio"].astype(float)
    out: dict[str, float | int] = {
        f"{prefix}_n": int(len(df)),
        f"{prefix}_mean": float(ratios.mean()) if len(df) else float("nan"),
        f"{prefix}_median": float(ratios.median()) if len(df) else float("nan"),
        f"{prefix}_std": float(ratios.std()) if len(df) > 1 else float("nan"),
        f"{prefix}_min": float(ratios.min()) if len(df) else float("nan"),
        f"{prefix}_max": float(ratios.max()) if len(df) else float("nan"),
        f"{prefix}_p10": float(ratios.quantile(0.10)) if len(df) else float("nan"),
        f"{prefix}_p90": float(ratios.quantile(0.90)) if len(df) else float("nan"),
        f"{prefix}_lt_01": int((ratios < 0.10).sum()) if len(df) else 0,
        f"{prefix}_lt_03": int((ratios < 0.30).sum()) if len(df) else 0,
        f"{prefix}_gt_09": int((ratios > 0.90).sum()) if len(df) else 0,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune tissue mask min_object_size/min_hole_size.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/preprocessing.yaml")
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Absolute/relative root containing avg3 cubes (patient_id/roi.npy).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/interim/masks/tuning",
        help="Directory where per-run masks and summary CSV are saved.",
    )
    parser.add_argument(
        "--object-sizes",
        type=str,
        default="300,800,1500,3000",
        help="Comma-separated grid for min_object_size.",
    )
    parser.add_argument(
        "--hole-sizes",
        type=str,
        default="300,800,1500,3000",
        help="Comma-separated grid for min_hole_size.",
    )
    parser.add_argument(
        "--max-rois",
        type=int,
        default=60,
        help="Max number of ROIs per grid run (use small number for fast tuning).",
    )
    parser.add_argument(
        "--tissue-side",
        type=str,
        choices=["auto", "dark", "bright"],
        default=None,
        help="Optional override for tissue side selection.",
    )
    parser.add_argument(
        "--target-tissue-ratio",
        type=float,
        default=None,
        help="Optional target ratio used when tissue_side=auto.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config).resolve())
    method = str(cfg["tissue_mask"]["method"])
    tissue_side = str(args.tissue_side or cfg["tissue_mask"].get("tissue_side", "auto"))
    target_tissue_ratio = args.target_tissue_ratio
    if target_tissue_ratio is None:
        raw_target = cfg["tissue_mask"].get("target_tissue_ratio", None)
        target_tissue_ratio = None if raw_target is None else float(raw_target)

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    object_sizes = _parse_int_list(args.object_sizes)
    hole_sizes = _parse_int_list(args.hole_sizes)

    rows: list[dict[str, float | int | str]] = []
    combos = list(product(object_sizes, hole_sizes))
    total = len(combos)

    for idx, (obj_size, hole_size) in enumerate(combos, start=1):
        run_name = f"obj{obj_size}_hole{hole_size}"
        run_out = output_root / run_name
        print(f"[{idx}/{total}] running {run_name}")

        manifest_df = build_masks_from_avg3(
            input_root=input_root,
            output_root=run_out,
            method=method,
            min_object_size=obj_size,
            min_hole_size=hole_size,
            tissue_side=tissue_side,
            target_tissue_ratio=target_tissue_ratio,
            max_rois=args.max_rois,
            verbose=args.verbose,
        )
        if manifest_df.empty:
            print(f"  -> no rows produced for {run_name}")
            continue

        local = manifest_df.copy()
        local["label_group"] = local["roi_name"].map(_label_from_roi)

        row: dict[str, float | int | str] = {
            "run_name": run_name,
            "min_object_size": obj_size,
            "min_hole_size": hole_size,
            "manifest_path": str((run_out / "mask_manifest.csv").resolve()),
            "tissue_side": tissue_side,
            "target_tissue_ratio": float(target_tissue_ratio) if target_tissue_ratio is not None else np.nan,
        }
        row.update(_ratio_stats(local, "all"))
        row.update(_ratio_stats(local[local["label_group"] == "T"], "T"))
        row.update(_ratio_stats(local[local["label_group"] == "NT"], "NT"))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved tuning summary: {summary_path}")
    if not summary_df.empty:
        print(summary_df[["run_name", "all_mean", "all_p10", "all_p90", "T_mean", "NT_mean"]])


if __name__ == "__main__":
    main()
