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


def compute_roi_metrics(
    cube_idxs: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Aggregate patch-level probabilities to ROI level and compute metrics.

    Each patch is identified by ``cube_idxs``.  All patches from the same cube
    are mean-pooled into a single ROI-level probability before metrics are
    computed.

    Args:
        cube_idxs: Per-patch cube indices, shape ``(N,)``.
        probs:     Per-patch probability for class 1, shape ``(N,)``.
        labels:    Per-patch ground-truth labels, shape ``(N,)``.

    Returns:
        Dict with: ``auc_roc_roi``, ``avg_precision_roi``, ``f1_at_0.5_roi``,
        ``f1_at_opt_roi``, ``threshold_opt_roi``, ``n_rois``.
        Values are ``float("nan")`` when the metric cannot be computed
        (e.g. only one class present in val set).
    """
    try:
        from sklearn.metrics import (  # type: ignore
            average_precision_score,
            f1_score,
            precision_recall_curve,
            roc_auc_score,
        )
    except ImportError:
        nan = float("nan")
        return {
            "auc_roc_roi": nan, "avg_precision_roi": nan,
            "f1_at_0.5_roi": nan, "f1_at_opt_roi": nan,
            "threshold_opt_roi": nan, "n_rois": 0,
        }

    cube_idxs = np.asarray(cube_idxs, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    unique_cubes = np.unique(cube_idxs)
    roi_probs = np.array([probs[cube_idxs == c].mean() for c in unique_cubes])
    roi_labels = np.array([labels[cube_idxs == c][0] for c in unique_cubes])

    n_rois = len(unique_cubes)
    nan = float("nan")

    if len(np.unique(roi_labels)) < 2:
        return {
            "auc_roc_roi": nan, "avg_precision_roi": nan,
            "f1_at_0.5_roi": nan, "f1_at_opt_roi": nan,
            "threshold_opt_roi": nan, "n_rois": n_rois,
        }

    auc_roc = float(roc_auc_score(roi_labels, roi_probs))
    avg_prec = float(average_precision_score(roi_labels, roi_probs))
    f1_05 = float(f1_score(roi_labels, (roi_probs >= 0.5).astype(int), zero_division=0))

    # F1 at optimal threshold (maximises F1 on this val set)
    precision_arr, recall_arr, thresholds = precision_recall_curve(roi_labels, roi_probs)
    f1_arr = 2 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-8)
    best_idx = int(np.argmax(f1_arr[:-1]))  # last element has no matching threshold
    f1_opt = float(f1_arr[best_idx])
    threshold_opt = float(thresholds[best_idx])

    return {
        "auc_roc_roi": auc_roc,
        "avg_precision_roi": avg_prec,
        "f1_at_0.5_roi": f1_05,
        "f1_at_opt_roi": f1_opt,
        "threshold_opt_roi": threshold_opt,
        "n_rois": n_rois,
    }


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
