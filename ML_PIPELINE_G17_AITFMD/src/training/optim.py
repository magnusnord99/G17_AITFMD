"""Optimizer and scheduler factory functions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def build_optimizer(
    params,
    cfg: dict,
) -> torch.optim.Optimizer:
    """Construct an optimizer from a config dict (train.yaml ``optimizer`` block).

    Supported names: ``adam`` (default), ``adamw``, ``sgd``.

    Args:
        params: Model parameters (``model.parameters()`` or param groups).
        cfg: Dict with keys ``lr``, ``weight_decay``, and optionally ``name``.

    Returns:
        Configured optimizer instance.
    """
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    name = str(cfg.get("name", "adam")).lower()

    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unknown optimizer name: {name!r}. Choose from adam, adamw, sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Construct a learning-rate scheduler from a config dict.

    Supported names: ``cosine``, ``step``.  Returns ``None`` when
    ``cfg["enabled"]`` is False or ``cfg["name"]`` is ``"none"``.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        cfg: Dict from train.yaml ``scheduler`` block.
        num_epochs: Total training epochs (used for cosine T_max).

    Returns:
        Scheduler instance, or None if scheduling is disabled.
    """
    if not cfg.get("enabled", False):
        return None

    name = str(cfg.get("name", "none")).lower()
    if name in ("none", ""):
        return None

    if name == "cosine":
        t_max = int(cfg.get("t_max_epochs", num_epochs))
        t_max = max(1, t_max)
        return CosineAnnealingLR(optimizer, T_max=t_max)

    if name == "step":
        step_size = int(cfg.get("step_size", 10))
        gamma = float(cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown scheduler name: {name!r}. Choose from cosine, step, none.")


def compute_class_weights(
    labels: list[int] | np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary classification.

    Uses the formula: ``weight_c = n_total / (2 * n_c)``

    Args:
        labels: 1-D array-like of integer class labels (0 or 1).
        device: Target device for the returned tensor.

    Returns:
        Float tensor of shape ``(2,)`` with weights for class 0 and class 1.
    """
    arr = np.asarray(labels, dtype=np.int64)
    n0 = float(np.sum(arr == 0)) or 1.0
    n1 = float(np.sum(arr == 1)) or 1.0
    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)
