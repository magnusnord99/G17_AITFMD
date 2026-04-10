"""Evaluation package: metrics, validation, and test reporting."""

from src.evaluation.aggregation import (
    majority_vote_aggregate,
    mean_prob_aggregate,
    patient_level_aggregate,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.inference import InferenceEngine, InferenceResult
from src.evaluation.metrics import (
    compute_eval_metrics,
    compute_pr,
    compute_roc,
    find_optimal_threshold,
)

__all__ = [
    "Evaluator",
    "InferenceEngine",
    "InferenceResult",
    "compute_eval_metrics",
    "compute_pr",
    "compute_roc",
    "find_optimal_threshold",
    "majority_vote_aggregate",
    "mean_prob_aggregate",
    "patient_level_aggregate",
]
