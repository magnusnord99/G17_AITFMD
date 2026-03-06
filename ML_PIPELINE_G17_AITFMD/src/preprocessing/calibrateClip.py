"""Calibration utilities for HistologyHSI-GB cubes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spectral


def load_envi_cube(hdr_path: Path, bin_path: Path) -> np.ndarray:
    """Load ENVI cube and return float32 array with shape (H, W, B)."""
    img = spectral.envi.open(str(hdr_path), str(bin_path))
    cube = np.asarray(img.load(), dtype=np.float32)
    return cube


def calibrate_cube(
    raw: np.ndarray,
    dark: np.ndarray,
    white: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Apply flat-field calibration."""
    calibrated = (raw - dark) / (white - dark + eps)
    return calibrated.astype(np.float32)

def clip_cube(cube: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Clip cube values to given range."""
    return np.clip(cube, clip_min, clip_max)

