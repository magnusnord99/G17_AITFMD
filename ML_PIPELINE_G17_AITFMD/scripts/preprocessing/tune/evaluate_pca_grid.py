"""Evaluate PCA explained variance for a small hyperparameter grid.

This script fits PCA on train pixels only and reports explained variance for
multiple values of `n_components`. It also records `output_dtype` options so
results can be compared against storage choices in one table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pca import flatten_cube


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (config_path.parent / p).resolve()


def _sample_train_pixels(
    train_rows: pd.DataFrame,
    input_root: Path,
    max_train_pixels: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    total = 0
    n_rows = len(train_rows)

    iterator = tqdm(train_rows.iterrows(), total=n_rows, desc="collect_train_pixels", unit="roi")
    for i, (_, row) in enumerate(iterator):
        cube_path = input_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        cube = np.load(cube_path).astype(np.float32, copy=False)
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
        raise RuntimeError("No train pixels collected for PCA grid evaluation.")

    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCA explained-variance grid.")
    parser.add_argument("--config", type=str, default="configs/preprocessing/pca.yaml")
    parser.add_argument(
        "--n-components",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="List of PCA n_components to evaluate.",
    )
    parser.add_argument(
        "--output-dtypes",
        type=str,
        nargs="+",
        default=["float16", "float32"],
        help="Output dtypes to include in table (for documentation).",
    )
    parser.add_argument(
        "--max-train-pixels",
        type=int,
        default=None,
        help="Override max_train_pixels from config.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/interim/pca_grid_results.csv",
        help="Path to save evaluation table.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    input_root = _resolve_path(config_path, cfg["paths"]["input_root"])
    split_csv = _resolve_path(config_path, cfg["paths"]["split_csv"])
    output_csv = (PROJECT_ROOT / args.output_csv).resolve()

    seed = int(cfg.get("seed", 42))
    svd_solver = str(cfg["pca"].get("svd_solver", "auto"))
    max_train_pixels = int(args.max_train_pixels or cfg["pca"]["max_train_pixels"])

    split_df = pd.read_csv(split_csv)
    train_rows = split_df[split_df["split"] == "train"].reset_index(drop=True)
    if train_rows.empty:
        raise RuntimeError("No train rows found in split CSV.")

    print(f"[grid] config: {config_path}")
    print(f"[grid] input_root: {input_root}")
    print(f"[grid] split_csv: {split_csv}")
    print(f"[grid] train rows: {len(train_rows)}")
    print(f"[grid] max_train_pixels: {max_train_pixels}")

    train_pixels = _sample_train_pixels(
        train_rows=train_rows,
        input_root=input_root,
        max_train_pixels=max_train_pixels,
        seed=seed,
    )
    print(f"[grid] collected train pixels: {train_pixels.shape}")

    rows: list[dict[str, int | str | float]] = []
    for n_components in args.n_components:
        pca = PCA(n_components=int(n_components), svd_solver=svd_solver, random_state=seed)
        pca.fit(train_pixels)
        evs = float(pca.explained_variance_ratio_.sum())

        for output_dtype in args.output_dtypes:
            rows.append(
                {
                    "n_components": int(n_components),
                    "output_dtype": str(output_dtype),
                    "explained_variance_sum": evs,
                    "seed": seed,
                    "max_train_pixels": max_train_pixels,
                    "svd_solver": svd_solver,
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["n_components", "output_dtype"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print("\n[grid] results:")
    print(result_df.to_string(index=False))
    print(f"\n[grid] saved: {output_csv}")


if __name__ == "__main__":
    main()
