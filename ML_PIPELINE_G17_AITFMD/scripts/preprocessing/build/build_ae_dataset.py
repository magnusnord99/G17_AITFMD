"""Build AE-transformed dataset from avg3 cubes using trained encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.autoencoder import ConvAutoencoder


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
    parser = argparse.ArgumentParser(description="Build AE-transformed cubes from avg3 dataset.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/autoencoder.yaml")
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
    model_dir = _resolve_path(config_path, paths["model_dir"])
    model_filename = str(paths["model_filename"])
    manifest_csv = _resolve_path(config_path, paths["manifest_csv"])

    model_path = model_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"AE model not found: {model_path}. Run train_autoencoder.py first."
        )

    ae_cfg = cfg["autoencoder"]
    in_channels = int(ae_cfg["in_channels"])
    latent_channels = int(ae_cfg["latent_channels"])

    output_dtype_name = str(cfg.get("build", {}).get("output_dtype", "float32"))
    output_dtype = _dtype_from_name(output_dtype_name)

    runtime_cfg = cfg.get("runtime", {})
    device_str = str(runtime_cfg.get("device", "cuda"))
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str in ("cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    verbose = bool(runtime_cfg.get("verbose", False))
    overwrite = bool(cfg.get("build", {}).get("overwrite", False))

    split_df = pd.read_csv(split_csv)
    split_df = _build_input_table(split_df, input_root)

    missing_total = int((~split_df["input_exists"]).sum())
    if missing_total > 0:
        preview = split_df.loc[~split_df["input_exists"], ["patient_id", "roi_name", "split", "input_path"]].head(10)
        print("[ae] missing input entries:")
        print(preview.to_string(index=False))
        raise FileNotFoundError("Some split entries do not have corresponding avg3 .npy files.")

    checkpoint = torch.load(model_path, map_location=device)
    model = ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[ae] config: {config_path}")
    print(f"[ae] input_root: {input_root}")
    print(f"[ae] output_root: {output_root}")
    print(f"[ae] model: {model_path} (latent_channels={latent_channels})")
    print(f"[ae] device: {device}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []

    iterator = split_df.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=len(split_df), desc="build_ae", unit="roi")

    written = 0
    with torch.no_grad():
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
                h, w, c = cube.shape
                x = torch.from_numpy(cube).permute(2, 0, 1).unsqueeze(0).to(device)
                z = model.encode(x)
                out_cube = z.squeeze(0).permute(1, 2, 0).cpu().numpy()
                np.save(out_path, out_cube.astype(output_dtype))
                status = "written"
                out_shape = "x".join(map(str, out_cube.shape))
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
    print(f"[ae] saved manifest: {manifest_csv} (rows={len(manifest_df)})")
    print(f"[ae] cubes written: {written}")


if __name__ == "__main__":
    main()
