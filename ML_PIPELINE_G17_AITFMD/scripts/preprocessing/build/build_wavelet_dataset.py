"""Build wavelet-reduced dataset from avg3 cubes."""

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

from src.preprocessing.wavelet import (
    reduce_cube_wavelet_1d,
    reduce_cube_wavelet_approx_detail_padded,
    reduce_cube_wavelet_approx_padded,
)


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
    parser = argparse.ArgumentParser(description="Build wavelet-reduced cubes from avg3 dataset.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/wavelet.yaml")
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional limit for number of ROI cubes written (smoke test).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    input_root = _resolve_path(config_path, cfg["paths"]["input_root"])
    output_root = _resolve_path(config_path, cfg["paths"]["output_root"])
    split_csv = _resolve_path(config_path, cfg["paths"]["split_csv"])
    manifest_csv = _resolve_path(config_path, cfg["paths"]["manifest_csv"])

    split_df = pd.read_csv(split_csv)
    split_df = _build_input_table(split_df, input_root)

    missing_total = int((~split_df["input_exists"]).sum())
    if missing_total > 0:
        preview = split_df.loc[~split_df["input_exists"], ["patient_id", "roi_name", "split", "input_path"]].head(10)
        print("[wavelet] missing input entries:")
        print(preview.to_string(index=False))
        raise FileNotFoundError("Some split entries do not have corresponding input .npy files.")

    wave_cfg = cfg["wavelet"]
    feature_mode = str(wave_cfg.get("feature_mode", "approx"))
    target_bands = int(wave_cfg.get("target_bands", 32))
    wavelet_name = str(wave_cfg.get("wavelet", "db2"))
    level = wave_cfg.get("level", None)
    if level is not None:
        level = int(level)
    mode = str(wave_cfg.get("mode", "symmetric"))
    pad_mode = str(wave_cfg.get("pad_mode", "edge"))
    compute_dtype = _dtype_from_name(str(wave_cfg.get("compute_dtype", "float32")))
    output_dtype = _dtype_from_name(str(wave_cfg.get("output_dtype", "float32")))

    runtime_cfg = cfg.get("runtime", {})
    verbose = bool(runtime_cfg.get("verbose", False))
    overwrite = bool(runtime_cfg.get("overwrite", False))

    print(f"[wavelet] config: {config_path}")
    print(f"[wavelet] input_root: {input_root}")
    print(f"[wavelet] output_root: {output_root}")
    print(f"[wavelet] split_csv: {split_csv}")
    print(f"[wavelet] target_bands: {target_bands}")
    print(f"[wavelet] feature_mode: {feature_mode}")
    print(f"[wavelet] wavelet: {wavelet_name} | level: {level} | mode: {mode} | pad_mode: {pad_mode}")
    print(f"[wavelet] dtypes compute={np.dtype(compute_dtype).name} output={np.dtype(output_dtype).name}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []

    iterator = split_df.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=len(split_df), desc="build_wavelet", unit="roi")

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
            cube = np.load(input_path).astype(compute_dtype, copy=False)
            if feature_mode == "approx":
                reduced = reduce_cube_wavelet_1d(
                    cube=cube.astype(np.float32, copy=False),
                    target_bands=target_bands,
                    wavelet=wavelet_name,
                    level=level,
                    mode=mode,
                )
            elif feature_mode == "approx_padded":
                reduced = reduce_cube_wavelet_approx_padded(
                    cube=cube.astype(np.float32, copy=False),
                    target_bands=target_bands,
                    wavelet=wavelet_name,
                    level=level,
                    mode=mode,
                    pad_mode=pad_mode,
                )
            elif feature_mode == "approx_detail":
                reduced = reduce_cube_wavelet_approx_detail_padded(
                    cube=cube.astype(np.float32, copy=False),
                    target_bands=target_bands,
                    wavelet=wavelet_name,
                    level=level,
                    mode=mode,
                    pad_mode=pad_mode,
                )
            else:
                raise ValueError(
                    f"Unsupported feature_mode: {feature_mode}. "
                    "Use 'approx', 'approx_padded', or 'approx_detail'."
                )
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
    print(f"[wavelet] saved manifest: {manifest_csv} (rows={len(manifest_df)})")
    print(f"[wavelet] cubes written: {written}")


if __name__ == "__main__":
    main()
