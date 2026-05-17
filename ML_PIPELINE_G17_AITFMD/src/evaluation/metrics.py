"""Evaluation-time metrics including AUC-ROC and Precision-Recall curves.

These functions operate on numpy arrays (post-inference) and can afford sklearn
calls.  They are intentionally separate from ``src.training.metrics`` which runs
on GPU tensors mid-loop.
"""

from __future__ import annotations

import numpy as np


def compute_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve.

    Returns:
        ``(fpr, tpr, thresholds)`` — arrays suitable for plotting and
        threshold selection.
    """
    from sklearn.metrics import roc_curve  # type: ignore

    return roc_curve(np.asarray(y_true), np.asarray(y_prob))


def compute_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve.

    Returns:
        ``(precision, recall, thresholds)`` — note sklearn's PR curve
        has ``len(thresholds) == len(precision) - 1``.
    """
    from sklearn.metrics import precision_recall_curve  # type: ignore

    return precision_recall_curve(np.asarray(y_true), np.asarray(y_prob))


def find_optimal_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    strategy: str = "youden",
) -> float:
    """Select the optimal probability threshold from an ROC curve.

    Args:
        fpr: False-positive rates from ``compute_roc``.
        tpr: True-positive rates from ``compute_roc``.
        thresholds: Thresholds from ``compute_roc``.
        strategy: ``"youden"`` maximises Youden's J (sensitivity + specificity - 1).

    Returns:
        Optimal threshold value.
    """
    if strategy == "youden":
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thresholds[idx])
    raise ValueError(f"Unknown threshold strategy: {strategy!r}")


def compute_eval_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Full evaluation metric set for one split.

    Computes binary metrics at the given threshold plus AUC-ROC and
    average precision (area under PR curve).

    Args:
        y_true: Ground-truth labels, shape ``(N,)``.
        y_prob: Probability for class 1, shape ``(N,)``.
        threshold: Decision threshold applied to probabilities.

    Returns:
        Dict with keys: ``threshold_used``, ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``tp``, ``tn``, ``fp``, ``fn``,
        ``auc_roc``, ``avg_precision``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

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
        "threshold_used": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if len(np.unique(y_true)) == 2:
        result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        result["avg_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        result["auc_roc"] = float("nan")
        result["avg_precision"] = float("nan")

    return result
