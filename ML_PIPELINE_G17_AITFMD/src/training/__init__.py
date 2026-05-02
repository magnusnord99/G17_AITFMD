"""Training package: loops, optimization, and checkpointing."""

from src.training.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from src.training.early_stopping import EarlyStopping
from src.training.metrics import compute_binary_metrics, format_metrics_table
from src.training.optim import build_optimizer, build_scheduler, compute_class_weights
from src.training.trainer import Trainer

__all__ = [
    "BaseCallback",
    "CheckpointCallback",
    "EarlyStopping",
    "EvalCallback",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "compute_binary_metrics",
    "compute_class_weights",
    "format_metrics_table",
]
