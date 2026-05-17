"""Early stopping callback for the training loop."""

from __future__ import annotations


class EarlyStopping:
    """Stopp trening når en overvåket metrikk slutter å forbedre seg.

    patience: antall epoker uten forbedring før stopp.
    min_delta: minimalt krav til forbedring.
    mode: "min" (lavere er bedre) eller "max" (høyere er bedre).
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
        """Oppdater teller og returner True om trening skal stoppes."""
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
        """Nullstill tilstand slik at instansen kan gjenbrukes."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.triggered = False
