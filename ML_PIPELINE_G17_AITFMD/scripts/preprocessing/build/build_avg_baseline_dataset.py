"""Build avg-baseline dataset: 275 bands -> 16 by simple averaging (no model)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.band_reduce import reduce_bands_by_avg
from src.preprocessing.masked_cube import prepare_cube_with_mask


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _dtype_from_name(name: str) -> np.dtype:
    allowed = {"float16": np.float16, "float32": np.float32}
    if name not in allowed:
        raise ValueError(f"Unsupported dtype '{name}'. Expected one of {sorted(allowed.keys())}.")
    return allowed[name]


def _build_input_table(split_df: pd.DataFrame, input_root: Path) -> pd.DataFrame:
    required_cols = {"patient_id", "roi_name", "split"}
    missing_cols = required_cols.difference(set(split_df.columns))
    if missing_cols:
        raise ValueError(f"Split CSV missing required columns: {sorted(missing_cols)}")

    split_df = split_df.copy()
    split_df["input_path"] = split_df.apply(
        lambda row: str(input_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"),
        axis=1,
    )
    split_df["input_exists"] = split_df["input_path"].map(lambda p: Path(p).exists())
    return split_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build avg-baseline cubes: 275->16 bands by averaging (no model fit)."
    )
    parser.add_argument("--config", type=str, default="configs/preprocessing/avg_baseline.yaml")
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional limit for number of transformed ROI cubes (smoke test).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config
    cfg = _load_yaml(config_path)

    paths = cfg["paths"]
    input_root = _resolve_path(config_path, paths["input_root"])
    output_root = _resolve_path(config_path, paths["output_root"])
    split_csv = _resolve_path(config_path, paths["split_csv"])
    manifest_csv = _resolve_path(config_path, paths["manifest_csv"])

    br_cfg = cfg["band_reduce"]
    in_bands = int(br_cfg["in_bands"])
    out_bands = int(br_cfg["out_bands"])
    strategy = str(br_cfg.get("strategy", "crop"))

    output_dtype = _dtype_from_name(str(cfg.get("build", {}).get("output_dtype", "float32")))
    overwrite = bool(cfg.get("build", {}).get("overwrite", False))
    verbose = bool(cfg.get("runtime", {}).get("verbose", True))

    mask_cfg = cfg.get("mask") or {}
    mask_root = None
    if mask_cfg.get("root"):
        mask_root = _resolve_path(config_path, str(mask_cfg["root"]))
    require_mask = bool(mask_cfg.get("require", False))
    apply_mask = bool(mask_cfg.get("apply_to_cube", True))
    if mask_root:
        print(
            f"[avg_baseline] mask: root={mask_root} require={require_mask} apply_to_cube={apply_mask}"
        )

    split_df = pd.read_csv(split_csv)
    split_df = _build_input_table(split_df, input_root)

    missing_total = int((~split_df["input_exists"]).sum())
    if missing_total > 0:
        preview = split_df.loc[~split_df["input_exists"], ["patient_id", "roi_name", "split", "input_path"]].head(10)
        print("[avg_baseline] missing input entries:")
        print(preview.to_string(index=False))
        raise FileNotFoundError("Some split entries do not have corresponding avg3 .npy files.")

    print(f"[avg_baseline] config: {config_path}")
    print(f"[avg_baseline] input_root: {input_root}")
    print(f"[avg_baseline] output_root: {output_root}")
    print(f"[avg_baseline] {in_bands} -> {out_bands} bands, strategy={strategy}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []

    iterator = split_df.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=len(split_df), desc="build_avg_baseline", unit="roi")

    written = 0
    for _, row in iterator:
        patient_id = str(row["patient_id"])
        roi_name = str(row["roi_name"])
        split = str(row["split"])
        input_path = Path(str(row["input_path"]))

        out_path = output_root / patient_id / f"{roi_name}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            status = "skipped_exists"
            out_shape = ""
        else:
            cube = np.load(input_path).astype(np.float32, copy=False)
            cube, _ = prepare_cube_with_mask(
                cube,
                mask_root,
                patient_id,
                roi_name,
                require_mask=require_mask,
                apply_to_cube=apply_mask,
            )
            if cube.shape[2] != in_bands:
                raise ValueError(
                    f"Cube {input_path} has {cube.shape[2]} bands, expected {in_bands}."
                )
            reduced = reduce_bands_by_avg(cube, n_out_bands=out_bands, strategy=strategy)
            np.save(out_path, reduced.astype(output_dtype))
            status = "written"
            out_shape = "x".join(map(str, reduced.shape))
            written += 1

        manifest_rows.append(
            {
                "patient_id": patient_id,
                "roi_name": roi_name,
                "split": split,
                "label_id": int(row["label_id"]) if "label_id" in row else -1,
                "input_path": str(input_path),
                "output_path": str(out_path),
                "output_shape": out_shape,
                "status": status,
            }
        )

        if args.max_rois is not None and written >= args.max_rois:
            break

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[avg_baseline] saved manifest: {manifest_csv} (rows={len(manifest_df)})")
    print(f"[avg_baseline] cubes written: {written}")


if __name__ == "__main__":
    main()
