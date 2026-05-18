"""Device selection for 3D CNN training and evaluation."""

from __future__ import annotations

import torch


def pick_training_device() -> torch.device:
    """Return the best device for 3D CNN workloads.

    MPS is skipped: Conv3D is not supported on Apple MPS in PyTorch 2.2.x (common on Mac).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
