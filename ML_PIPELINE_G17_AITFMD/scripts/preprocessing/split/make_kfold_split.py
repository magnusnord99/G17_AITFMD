"""
Assign patient-level k-fold splits to a manifest CSV.

Adds a `fold` column (0..k-1) using round-robin assignment over
sorted patient IDs, with mixed-label patients (those with both T and NT
ROIs) interleaved with tumor-only patients so that each fold has at
least one patient with NT tissue.

Output: data/interim/wavelet_manifest_kfold{k}.csv

Usage:
    python scripts/preprocessing/split/make_kfold_split.py
    python scripts/preprocessing/split/make_kfold_split.py --k 5
    python scripts/preprocessing/split/make_kfold_split.py --manifest data/interim/wavelet_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add patient-level k-fold column to manifest.")
    p.add_argument("--manifest", default="data/interim/wavelet_manifest.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--output", default=None, help="Output path (default: auto)")
    p.add_argument("--seed", type=int, default=42, help="Unused (assignment is deterministic)")
    return p.parse_args()


def assign_folds(patient_ids: list[str], labels_per_patient: dict[str, set[int]], k: int) -> dict[str, int]:
    """
    Assign fold IDs to patients using stratified round-robin.

    Mixed patients (have both label 0 and 1) are assigned first in
    sorted order, then tumor-only patients are interleaved so each
    fold receives at least one mixed patient where possible.
    """
    mixed = sorted(p for p, lbls in labels_per_patient.items() if len(lbls) > 1)
    single = sorted(p for p, lbls in labels_per_patient.items() if len(lbls) == 1)

    assignment: dict[str, int] = {}
    for i, p in enumerate(mixed):
        assignment[p] = i % k
    for i, p in enumerate(single):
        assignment[p] = i % k

    return assignment


def main() -> int:
    args = _parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    if not manifest_path.exists():
        manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(manifest_path)
    required = {"patient_id", "label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    k = args.k
    patient_ids = sorted(df["patient_id"].unique())
    labels_per_patient: dict[str, set[int]] = {
        p: set(df[df["patient_id"] == p]["label_id"].tolist())
        for p in patient_ids
    }

    fold_map = assign_folds(patient_ids, labels_per_patient, k)
    df["fold"] = df["patient_id"].map(fold_map)

    # Print summary
    print(f"\nK-fold split (k={k}) for {len(patient_ids)} patients\n")
    print(f"{'Fold':<6} {'Patients':<40} {'n_rois':<8} {'labels'}")
    print("-" * 70)
    for fold_idx in range(k):
        fold_df = df[df["fold"] == fold_idx]
        patients_in_fold = sorted(fold_df["patient_id"].unique())
        labels = sorted(fold_df["label_id"].unique())
        print(f"{fold_idx:<6} {', '.join(patients_in_fold):<40} {len(fold_df):<8} {labels}")

    print()
    # Validate: each fold must have both labels to compute AUC
    for fold_idx in range(k):
        fold_df = df[df["fold"] == fold_idx]
        present = set(fold_df["label_id"].unique())
        if len(present) < 2:
            print(f"WARNING: fold {fold_idx} has only label(s) {present} — AUC will be undefined for this test fold.")

    out_path: Path
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = manifest_path.parent / f"{manifest_path.stem}_kfold{k}.csv"

    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
