"""Dataset indexing for HistologyHSI-GB ENVI folder structure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FILES = {
    "raw_path": "raw",
    "raw_hdr_path": "raw.hdr",
    "dark_path": "darkReference",
    "dark_hdr_path": "darkReference.hdr",
    "white_path": "whiteReference",
    "white_hdr_path": "whiteReference.hdr",
}


def _is_ignored_name(name: str) -> bool:
    """Ignore AppleDouble/hidden artifacts from macOS."""
    return name.startswith("._") or name.startswith(".")


def _parse_label(roi_name: str, tumor_suffix: str, non_tumor_suffix: str) -> tuple[str, int]:
    """Extract class label from ROI folder name."""
    if roi_name.endswith(non_tumor_suffix):
        return "NT", 0
    if roi_name.endswith(tumor_suffix):
        return "T", 1
    return "UNKNOWN", -1


def _candidate_patient_dirs(dataset_root: Path) -> list[Path]:
    """Return sorted patient directories (P1, P2, ...)."""
    candidates = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not _is_ignored_name(path.name) and path.name.startswith("P")
    ]
    return sorted(candidates, key=lambda p: p.name)


def build_dataset_index(
    dataset_root: Path,
    tumor_suffix: str = "_T",
    non_tumor_suffix: str = "_NT",
) -> pd.DataFrame:
    """
    Build metadata table with one row per ROI folder.

    Columns include patient/ROI ids, class labels, file paths, missing files,
    and an `is_valid` flag indicating whether required files and label are valid.
    """
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    rows: list[dict[str, Any]] = []
    for patient_dir in _candidate_patient_dirs(dataset_root):
        roi_dirs = [
            path
            for path in patient_dir.iterdir()
            if path.is_dir() and not _is_ignored_name(path.name)
        ]

        for roi_dir in sorted(roi_dirs, key=lambda p: p.name):
            label_str, label_id = _parse_label(roi_dir.name, tumor_suffix, non_tumor_suffix)
            required_paths = {key: (roi_dir / rel_name) for key, rel_name in REQUIRED_FILES.items()}
            missing = [key for key, path in required_paths.items() if not path.exists()]

            rgb_path = roi_dir / "rgb.png"
            row = {
                "patient_id": patient_dir.name,
                "roi_name": roi_dir.name,
                "label_str": label_str,
                "label_id": label_id,
                "roi_path": str(roi_dir),
                "rgb_path": str(rgb_path),
                "rgb_exists": rgb_path.exists(),
                "missing_files": ",".join(missing),
                "is_valid": bool(label_id != -1 and not missing),
            }
            row.update({key: str(path) for key, path in required_paths.items()})
            row.update({f"{key.replace('_path', '')}_exists": path.exists() for key, path in required_paths.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["patient_id", "roi_name"]).reset_index(drop=True)
    return df


def save_dataset_index(df: pd.DataFrame, metadata_csv_path: Path) -> Path:
    """Persist dataset index to CSV and return final output path."""
    metadata_csv_path = metadata_csv_path.resolve()
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_csv_path, index=False)
    return metadata_csv_path


def summarize_index(df: pd.DataFrame) -> dict[str, int]:
    """Compute quick summary stats for terminal output."""
    if df.empty:
        return {
            "num_patients": 0,
            "num_rois_total": 0,
            "num_rois_valid": 0,
            "num_rois_invalid": 0,
            "num_tumor": 0,
            "num_non_tumor": 0,
            "num_unknown_label": 0,
        }

    return {
        "num_patients": int(df["patient_id"].nunique()),
        "num_rois_total": int(len(df)),
        "num_rois_valid": int(df["is_valid"].sum()),
        "num_rois_invalid": int((~df["is_valid"]).sum()),
        "num_tumor": int((df["label_str"] == "T").sum()),
        "num_non_tumor": int((df["label_str"] == "NT").sum()),
        "num_unknown_label": int((df["label_id"] == -1).sum()),
    }

