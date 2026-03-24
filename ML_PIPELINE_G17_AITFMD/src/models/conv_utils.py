"""Felles hjelpere for Conv3d (HSI: spektral D × H × W)."""

from __future__ import annotations

from typing import List, Tuple, Union

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
