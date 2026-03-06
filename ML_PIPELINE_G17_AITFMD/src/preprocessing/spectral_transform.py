"""Spectral transform utilities for hyperspectral cubes."""

from __future__ import annotations

import numpy as np


def reduce_bands_neighbor_average(cube: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Reduce spectral bands by averaging adjacent neighbors.
    """
    if window <= 1:
        return cube

    h, w, b = cube.shape
    b_reduced = b // window
    if b_reduced == 0:
        raise ValueError(f"window={window} is too large for b={b}")

    trimmed = cube[:, :, : b_reduced * window]
    reshaped = trimmed.reshape(h, w, b_reduced, window)
    reduced = reshaped.mean(axis=3)
    return reduced.astype(np.float32)