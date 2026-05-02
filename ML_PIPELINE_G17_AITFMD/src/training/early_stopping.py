"""Early stopping callback for the training loop."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` (lower is better, e.g. val_loss) or
              ``"max"`` (higher is better, e.g. val_f1).
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode

        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.triggered: bool = False

    def __call__(self, metric_value: float) -> bool:
        """Evaluate the metric and return True when training should stop."""
        if self.mode == "min":
            improved = metric_value < self.best_value - self.min_delta
        else:
            improved = metric_value > self.best_value + self.min_delta

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False

    def reset(self) -> None:
        """Reset state so the instance can be reused."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.triggered = False
