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
    """Bygg optimizer fra config-dict (train.yaml optimizer-blokk).

    Støttede navn: adam (default), adamw, sgd.
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
    """Bygg LR-scheduler fra config-dict. Returner None om scheduling er deaktivert.

    Støttede navn: cosine, step.
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
    """Beregn inverse-frekvens klasseveiere: weight_c = n_total / (2 * n_c).

    Returner float-tensor med form (2,) for klasse 0 og 1.
    """
    arr = np.asarray(labels, dtype=np.int64)
    n0 = float(np.sum(arr == 0)) or 1.0
    n1 = float(np.sum(arr == 1)) or 1.0
    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)
