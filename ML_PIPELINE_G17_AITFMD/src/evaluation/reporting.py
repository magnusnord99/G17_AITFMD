"""Output helpers: JSON reports, ROC/PR plots, eval summary CSV."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def save_eval_json(
    per_roi: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Skriv evalueringsrapport til JSON-fil og returner faktisk filsti."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        report.update(metadata)
    report["roi_metrics"] = metrics
    report["per_roi"] = per_roi

    output_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    log.info("Eval report written: %s", output_path)
    return output_path.resolve()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: Path,
    title: str = "ROC Curve",
) -> None:
    """Lagre ROC-kurve som PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="C0", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve saved: %s", output_path)


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    output_path: Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """Lagre presisjon-tilbakekall-kurve som PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color="C1", lw=2, label=f"AP = {avg_precision:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PR curve saved: %s", output_path)


def append_eval_summary_csv(
    metrics: dict[str, Any],
    csv_path: Path,
    run_id: str,
    split: str,
) -> None:
    """Legg til én rad i persistent eval-CSV. Oppretter fil med header om den ikke finnes."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_metrics(metrics)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "split": split,
        **flat,
    }

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    log.info("Eval summary appended: %s", csv_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return str(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _flatten_metrics(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested metrics dict for CSV storage."""
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=key + "."))
        elif isinstance(v, (int, float, str, bool)):
            flat[key] = v
        else:
            flat[key] = json.dumps(v, default=_json_default)
    return flat
