"""Tissue mask utilities for hyperspectral cubes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects


def _validate_cube(cube: np.ndarray) -> None:
    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H, W, B), got {cube.shape}")


def _mean_intensity_map(cube: np.ndarray) -> np.ndarray:
    _validate_cube(cube)
    return cube.mean(axis=2, dtype=np.float32)

def _std_intensity_map(cube: np.ndarray) -> np.ndarray:
    _validate_cube(cube)
    return cube.std(axis=2, dtype=np.float32)


def _clean_mask(mask: np.ndarray, min_object_size: int, min_hole_size: int) -> np.ndarray:
    # skimage>=0.26 renamed min_size/area_threshold -> max_size with
    # slightly different boundary semantics (<=). We keep previous behavior
    # by subtracting one from the threshold.
    object_threshold = max(0, min_object_size - 1)
    hole_threshold = max(0, min_hole_size - 1)
    try:
        cleaned = remove_small_objects(mask, max_size=object_threshold)
    except TypeError:
        cleaned = remove_small_objects(mask, min_size=min_object_size)
    try:
        cleaned = remove_small_holes(cleaned, max_size=hole_threshold)
    except TypeError:
        cleaned = remove_small_holes(cleaned, area_threshold=min_hole_size)
    return cleaned.astype(bool)


def build_tissue_mask(
    cube: np.ndarray,
    method: str = "mean_otsu",
    min_object_size: int = 1000,
    min_hole_size: int = 1000,
    tissue_side: str = "auto",
    target_tissue_ratio: float | None = None,
    q_mean: float = 0.50,
    q_std: float = 0.40,
) -> np.ndarray:
    """
    Build boolean tissue mask from one hyperspectral cube.

    Supported methods:
      - mean_otsu:
          threshold mean intensity image with Otsu, then pick tissue side
          as configured (dark / bright / auto).
      - mean_std_percentile:
          build tissue mask from two per-pixel features:
            * mean intensity (mu)
            * spectral std over bands (sigma)
          tissue = (sigma > quantile(sigma, q_std)) OR
                   (mu < quantile(mu, q_mean))
    """
    if method not in {"mean_otsu", "mean_std_percentile"}:
        raise ValueError(f"Unsupported tissue mask method: {method}")
    if tissue_side not in {"auto", "dark", "bright"}:
        raise ValueError("tissue_side must be one of: {'auto', 'dark', 'bright'}")
    if target_tissue_ratio is not None and not (0.0 < target_tissue_ratio < 1.0):
        raise ValueError(f"target_tissue_ratio must be in (0, 1), got {target_tissue_ratio}")
    if not (0.0 < q_mean < 1.0):
        raise ValueError(f"q_mean must be in (0, 1), got {q_mean}")
    if not (0.0 < q_std < 1.0):
        raise ValueError(f"q_std must be in (0, 1), got {q_std}")

    gray = _mean_intensity_map(cube)
    if method == "mean_std_percentile":
        std = _std_intensity_map(cube)
        t_mean = float(np.quantile(gray, q_mean))
        t_std = float(np.quantile(std, q_std))
        raw_mask = (std > t_std) | (gray < t_mean)
        return _clean_mask(raw_mask, min_object_size=min_object_size, min_hole_size=min_hole_size)

    threshold = threshold_otsu(gray)
    low_mask = gray <= threshold
    high_mask = gray > threshold

    low_clean = _clean_mask(low_mask, min_object_size=min_object_size, min_hole_size=min_hole_size)
    high_clean = _clean_mask(high_mask, min_object_size=min_object_size, min_hole_size=min_hole_size)

    if tissue_side == "dark":
        return low_clean
    if tissue_side == "bright":
        return high_clean

    if target_tissue_ratio is not None:
        low_delta = abs(float(low_clean.mean()) - float(target_tissue_ratio))
        high_delta = abs(float(high_clean.mean()) - float(target_tissue_ratio))
        return low_clean if low_delta <= high_delta else high_clean

    # Backward-compatible fallback: use darker side as tissue.
    low_mean = float(gray[low_mask].mean()) if low_mask.any() else np.inf
    high_mean = float(gray[high_mask].mean()) if high_mask.any() else np.inf
    return low_clean if low_mean <= high_mean else high_clean


def tissue_ratio(mask: np.ndarray) -> float:
    """Return tissue fraction in [0, 1]."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    return float(mask.mean())


def save_tissue_mask(mask: np.ndarray, out_path: Path) -> Path:
    """Persist mask as .npy with uint8 values {0,1}."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mask.astype(np.uint8))
    return out_path
