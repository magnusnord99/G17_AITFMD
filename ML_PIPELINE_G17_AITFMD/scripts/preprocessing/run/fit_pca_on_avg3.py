"""Fit PCA on train avg3 cubes and transform all splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pca import fit_pca_from_pixels, flatten_cube, transform_cube_with_pca


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(config_path: Path, raw_path: str) -> Path:
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


def _sample_train_pixels(
    train_rows: pd.DataFrame,
    max_train_pixels: int,
    compute_dtype: np.dtype,
    seed: int,
    verbose: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    total = 0
    n_rows = len(train_rows)

    iterator = train_rows.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=n_rows, desc="collect_train_pixels", unit="roi")

    for i, (_, row) in enumerate(iterator):
        cube = np.load(Path(row["input_path"])).astype(compute_dtype, copy=False)
        flat = flatten_cube(cube)

        remaining_budget = max_train_pixels - total
        remaining_rois = max(1, n_rows - i)
        target_this_roi = max(1, remaining_budget // remaining_rois)
        take = min(len(flat), target_this_roi, remaining_budget)
        if take <= 0:
            break

        if take < len(flat):
            idx = rng.choice(len(flat), size=take, replace=False)
            flat = flat[idx]
        parts.append(flat)
        total += take

        if total >= max_train_pixels:
            break

    if not parts:
        raise RuntimeError("No train pixels were collected for PCA fitting.")

    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA on avg3 train split and transform all splits.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/pca.yaml")
    parser.add_argument(
        "--max-rois",
        type=int,
        default=None,
        help="Optional limit for number of transformed ROI cubes (smoke test).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    input_root = resolve_path(config_path, cfg["paths"]["input_root"])
    output_root = resolve_path(config_path, cfg["paths"]["output_root"])
    split_csv = resolve_path(config_path, cfg["paths"]["split_csv"])
    model_dir = resolve_path(config_path, cfg["paths"]["model_dir"])
    model_filename = str(cfg["paths"]["model_filename"])
    manifest_csv = resolve_path(config_path, cfg["paths"]["manifest_csv"])

    split_df = pd.read_csv(split_csv)
    split_df = _build_input_table(split_df, input_root)

    missing_total = int((~split_df["input_exists"]).sum())
    missing_by_split = (
        split_df.loc[~split_df["input_exists"], "split"]
        .value_counts(dropna=False)
        .to_dict()
    )

    print(f"[pca] config: {config_path}")
    print(f"[pca] input_root: {input_root}")
    print(f"[pca] output_root: {output_root}")
    print(f"[pca] split_csv: {split_csv}")
    print(f"[pca] model_dir: {model_dir}")
    print(f"[pca] model_filename: {model_filename}")
    print(f"[pca] manifest_csv: {manifest_csv}")
    print(f"[pca] split rows: {len(split_df)}")
    print(f"[pca] split counts: {split_df['split'].value_counts(dropna=False).to_dict()}")
    print(f"[pca] missing input files: {missing_total}")
    if missing_total > 0:
        print(f"[pca] missing by split: {missing_by_split}")
        preview = split_df.loc[~split_df["input_exists"], ["patient_id", "roi_name", "split", "input_path"]].head(10)
        print("[pca] first missing entries:")
        print(preview.to_string(index=False))
        raise FileNotFoundError("Some split entries do not have corresponding avg3 .npy files.")

    pca_cfg = cfg["pca"]
    seed = int(cfg.get("seed", 42))
    n_components = int(pca_cfg["n_components"])
    max_train_pixels = int(pca_cfg["max_train_pixels"])
    svd_solver = str(pca_cfg.get("svd_solver", "auto"))
    compute_dtype = _dtype_from_name(str(pca_cfg.get("compute_dtype", "float32")))
    output_dtype = _dtype_from_name(str(pca_cfg.get("output_dtype", "float32")))

    runtime_cfg = cfg.get("runtime", {})
    verbose = bool(runtime_cfg.get("verbose", False))
    overwrite = bool(runtime_cfg.get("overwrite", False))

    train_rows = split_df[split_df["split"] == "train"].reset_index(drop=True)
    if train_rows.empty:
        raise RuntimeError("No train rows found in split CSV.")

    train_rows = train_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[pca] collecting train pixels (max={max_train_pixels}) ...")
    train_pixels = _sample_train_pixels(
        train_rows=train_rows,
        max_train_pixels=max_train_pixels,
        compute_dtype=compute_dtype,
        seed=seed,
        verbose=verbose,
    )
    print(f"[pca] collected train pixels: {train_pixels.shape}")

    print(f"[pca] fitting PCA (n_components={n_components}, svd_solver={svd_solver}) ...")
    pca_model = fit_pca_from_pixels(
        train_pixels=train_pixels,
        n_components=n_components,
        random_state=seed,
        svd_solver=svd_solver,
    )
    print(f"[pca] PCA fitted. explained_variance_sum={float(pca_model.explained_variance_ratio_.sum()):.6f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename
    dump(pca_model, model_path)
    print(f"[pca] saved model: {model_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []

    iterator = split_df.iterrows()
    if not verbose:
        iterator = tqdm(iterator, total=len(split_df), desc="transform_pca", unit="roi")

    transformed = 0
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
            transformed_cube = transform_cube_with_pca(cube.astype(np.float32, copy=False), pca_model)
            np.save(out_path, transformed_cube.astype(output_dtype))
            status = "written"
            out_shape = "x".join(map(str, transformed_cube.shape))
            transformed += 1

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

        if args.max_rois is not None and transformed >= args.max_rois:
            break

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[pca] saved manifest: {manifest_csv} (rows={len(manifest_df)})")
    print(f"[pca] transformed cubes written: {transformed}")


if __name__ == "__main__":
    main()
