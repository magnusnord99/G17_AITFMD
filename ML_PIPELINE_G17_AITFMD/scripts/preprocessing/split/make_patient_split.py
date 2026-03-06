"""Create patient-level train/val/test split CSV with class-safety checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _ratio_to_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if (ratios <= 0).any():
        raise ValueError("All split ratios must be > 0.")

    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())

    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[idx] += 1

    train_n, val_n, test_n = map(int, counts.tolist())

    # Keep each split non-empty when possible.
    if n >= 3:
        if val_n == 0:
            val_n = 1
            train_n -= 1
        if test_n == 0:
            test_n = 1
            train_n -= 1
        if train_n <= 0:
            raise ValueError("Ratios are too aggressive for this number of patients.")

    return train_n, val_n, test_n


def _split_has_both_classes(df: pd.DataFrame, patient_ids: set[str]) -> bool:
    labels = set(df[df["patient_id"].isin(patient_ids)]["label_id"].astype(int).unique().tolist())
    return 0 in labels and 1 in labels


def _find_valid_patient_split(
    df: pd.DataFrame,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    max_attempts: int,
) -> dict[str, set[str]]:
    patients = sorted(df["patient_id"].astype(str).unique().tolist())
    if train_n + val_n + test_n != len(patients):
        raise ValueError("Split counts do not sum to number of patients.")

    patient_arr = np.array(patients, dtype=object)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        shuffled = patient_arr.copy()
        rng.shuffle(shuffled)

        train_ids = set(shuffled[:train_n].tolist())
        val_ids = set(shuffled[train_n : train_n + val_n].tolist())
        test_ids = set(shuffled[train_n + val_n :].tolist())

        if _split_has_both_classes(df, train_ids) and _split_has_both_classes(df, val_ids) and _split_has_both_classes(df, test_ids):
            return {"train": train_ids, "val": val_ids, "test": test_ids}

    raise RuntimeError(
        f"Could not find a class-safe patient split after {max_attempts} attempts. "
        "Try increasing attempts or adjusting split ratios."
    )


def _print_summary(split_df: pd.DataFrame) -> None:
    print("\nSplit summary (patients / ROI):")
    for split_name in ["train", "val", "test"]:
        sub = split_df[split_df["split"] == split_name]
        n_pat = int(sub["patient_id"].nunique())
        n_roi = int(len(sub))
        n_t = int((sub["label_id"] == 1).sum())
        n_nt = int((sub["label_id"] == 0).sum())
        print(f"  {split_name:5s} patients={n_pat:2d} rois={n_roi:3d} tumor={n_t:3d} non_tumor={n_nt:3d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test split.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to train config YAML.")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="data/interim/metadata_master.csv",
        help="Path to metadata CSV from indexing.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/splits/patient_split.csv",
        help="Output split CSV path.",
    )
    parser.add_argument("--train-ratio", type=float, default=None, help="Train ratio (overrides config).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation ratio (overrides config).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test ratio (overrides config).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    parser.add_argument("--max-attempts", type=int, default=5000, help="Max random attempts for class-safe split.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    default_train = float(cfg.get("splits", {}).get("train_ratio", 0.70))
    default_val = float(cfg.get("splits", {}).get("val_ratio", 0.15))
    default_test = float(cfg.get("splits", {}).get("test_ratio", 0.15))
    default_seed = int(cfg.get("seed", 42))

    train_ratio = float(args.train_ratio if args.train_ratio is not None else default_train)
    val_ratio = float(args.val_ratio if args.val_ratio is not None else default_val)
    test_ratio = float(args.test_ratio if args.test_ratio is not None else default_test)
    seed = int(args.seed if args.seed is not None else default_seed)

    metadata_csv = (PROJECT_ROOT / args.metadata_csv).resolve()
    output_csv = (PROJECT_ROOT / args.output_csv).resolve()

    df = pd.read_csv(metadata_csv)
    if "is_valid" in df.columns:
        df = df[df["is_valid"]].copy()
    df = df[df["label_id"].isin([0, 1])].copy()

    n_patients = int(df["patient_id"].nunique())
    train_n, val_n, test_n = _ratio_to_counts(n_patients, train_ratio, val_ratio, test_ratio)
    print(
        f"Using ratios train/val/test={train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f} "
        f"-> patient counts {train_n}/{val_n}/{test_n} (n={n_patients})"
    )

    split_patients = _find_valid_patient_split(
        df=df,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        seed=seed,
        max_attempts=args.max_attempts,
    )

    patient_to_split = {}
    for split_name, ids in split_patients.items():
        for patient_id in ids:
            patient_to_split[patient_id] = split_name

    split_df = df.copy()
    split_df["split"] = split_df["patient_id"].map(patient_to_split)

    keep_cols = [
        "patient_id",
        "roi_name",
        "label_str",
        "label_id",
        "split",
        "raw_path",
        "raw_hdr_path",
        "dark_path",
        "dark_hdr_path",
        "white_path",
        "white_hdr_path",
    ]
    keep_cols = [col for col in keep_cols if col in split_df.columns]
    split_df = split_df[keep_cols].sort_values(["split", "patient_id", "roi_name"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    print(f"Saved split CSV: {output_csv}")
    _print_summary(split_df)


if __name__ == "__main__":
    main()
