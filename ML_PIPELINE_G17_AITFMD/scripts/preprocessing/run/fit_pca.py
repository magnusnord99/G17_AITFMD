"""Fit PCA on train avg3 cubes and save model."""

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

from src.preprocessing.masked_cube import load_binary_mask, mask_path_for_roi
from src.preprocessing.pca import fit_pca_from_pixels, flatten_cube


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


def _sample_train_pixels(
    train_rows: pd.DataFrame,
    max_train_pixels: int,
    compute_dtype: np.dtype,
    seed: int,
    verbose: bool,
    mask_root: Path | None = None,
    require_mask: bool = False,
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
        flat = flatten_cube(cube.astype(np.float32, copy=False))

        if mask_root is not None:
            mp = mask_path_for_roi(mask_root, str(row["patient_id"]), str(row["roi_name"]))
            if not mp.exists():
                if require_mask:
                    raise FileNotFoundError(
                        f"PCA fit require_mask=True but mask missing: {mp}"
                    )
            else:
                mask = load_binary_mask(mp)
                if mask.shape != cube.shape[:2]:
                    raise ValueError(
                        f"Mask {mp.shape} != cube spatial {cube.shape[:2]}"
                    )
                mflat = mask.reshape(-1)
                flat = flat[mflat > 0]
                if flat.size == 0:
                    continue

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
    parser = argparse.ArgumentParser(description="Fit PCA on avg3 train split and save model.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/pca.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config
    cfg = _load_yaml(config_path)

    paths = cfg["paths"]
    input_root = _resolve_path(config_path, paths["input_root"])
    split_csv = _resolve_path(config_path, paths["split_csv"])
    model_dir = _resolve_path(config_path, paths["model_dir"])
    model_filename = str(paths["model_filename"])

    split_df = pd.read_csv(split_csv)
    split_df = _build_input_table(split_df, input_root)

    missing_total = int((~split_df["input_exists"]).sum())
    if missing_total > 0:
        preview = split_df.loc[~split_df["input_exists"], ["patient_id", "roi_name", "split", "input_path"]].head(10)
        print("[pca] missing input entries:")
        print(preview.to_string(index=False))
        raise FileNotFoundError("Some split entries do not have corresponding avg3 .npy files.")

    pca_cfg = cfg["pca"]
    seed = int(cfg.get("seed", 42))
    n_components = int(pca_cfg["n_components"])
    max_train_pixels = int(pca_cfg["max_train_pixels"])
    svd_solver = str(pca_cfg.get("svd_solver", "auto"))
    compute_dtype = _dtype_from_name(str(pca_cfg.get("compute_dtype", "float32")))

    runtime_cfg = cfg.get("runtime", {})
    verbose = bool(runtime_cfg.get("verbose", False))

    mask_cfg = cfg.get("mask") or {}
    mask_root = None
    if mask_cfg.get("root"):
        mask_root = _resolve_path(config_path, str(mask_cfg["root"]))
    require_mask = bool(mask_cfg.get("require", False))
    if mask_root:
        print(
            f"[pca fit] mask: root={mask_root} require={require_mask} "
            "(only tissue pixels used for fitting)"
        )

    train_rows = split_df[split_df["split"] == "train"].reset_index(drop=True)
    if train_rows.empty:
        raise RuntimeError("No train rows found in split CSV.")

    train_rows = train_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[pca] config: {config_path}")
    print(f"[pca] input_root: {input_root}")
    print(f"[pca] model_dir: {model_dir}")
    print(f"[pca] collecting train pixels (max={max_train_pixels}) ...")

    train_pixels = _sample_train_pixels(
        train_rows=train_rows,
        max_train_pixels=max_train_pixels,
        compute_dtype=compute_dtype,
        seed=seed,
        verbose=verbose,
        mask_root=mask_root,
        require_mask=require_mask,
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


if __name__ == "__main__":
    main()
