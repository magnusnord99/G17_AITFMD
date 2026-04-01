"""Felles hjelpere for Conv3d (HSI: spektral D × H × W)."""

from __future__ import annotations

from typing import List, Tuple, Union

import torch.nn as nn

Kernel3D = Union[int, Tuple[int, int, int], List[int]]


def normalize_conv3d_kernel_size(kernel_size: Kernel3D) -> tuple[int, int, int]:
    """
    Gjør `kernel_size` om til (k_D, k_H, k_W) for nn.Conv3d.

    - int k → kubisk k×k×k
    - sekvens med tre tall → (k_D, k_H, k_W)

    Bruk med ``padding=0`` (valid) eller egen padding etter behov.
    """
    if isinstance(kernel_size, int):
        k = int(kernel_size)
        return (k, k, k)
    seq = tuple(int(x) for x in kernel_size)
    if len(seq) != 3:
        raise ValueError(
            f"kernel_size must be an int or a length-3 sequence (D,H,W), got {kernel_size!r}"
        )
    return seq


def make_groupnorm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm med automatisk tilpasset antall grupper.

    Finner det største antallet grupper ≤ max_groups som deler channels.
    Sikkert selv for uvanlige kanalantall (f.eks. etter concatenation).
    """
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def same_pad(kernel_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Beregn same-padding for odde kernel size (k//2 per dimensjon)."""
    return tuple(k // 2 for k in kernel_size)  # type: ignore[return-value]
