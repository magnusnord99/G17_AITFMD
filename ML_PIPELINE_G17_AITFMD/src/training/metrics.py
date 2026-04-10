"""Binary classification metrics for use during training.

These are pure numpy/sklearn functions that operate on CPU arrays and can be
called from both the training loop and the evaluation pipeline.
"""

from __future__ import annotations

import numpy as np


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute binary classification metrics from hard predictions.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``, values 0 or 1.
        y_pred: Hard predicted labels, shape ``(N,)``, values 0 or 1.
        y_prob: Soft probability for class 1, shape ``(N,)``.  If provided,
                ``auc_roc`` is added to the output dict (requires sklearn).

    Returns:
        Dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``tp``, ``tn``, ``fp``, ``fn``, and optionally ``auc_roc``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / total

    result: dict[str, float | int] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            y_prob = np.asarray(y_prob, dtype=np.float32)
            if len(np.unique(y_true)) == 2:
                result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ImportError:
            pass

    return result


def format_metrics_table(metrics: dict[str, float | int]) -> str:
    """Format a metrics dict as a single INFO-level log line."""
    parts = []
    order = ["accuracy", "precision", "recall", "f1", "auc_roc", "tp", "tn", "fp", "fn"]
    for key in order:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            parts.append(f"{key}={val:.4f}")
        else:
            parts.append(f"{key}={val}")
    return "  ".join(parts)
