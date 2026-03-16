"""
Dummy 3D CNN backend for GUI testing.

Returns pseudo-random "cancer probability" per patch so the GUI can display
a heatmap before the real model is trained.
"""

from __future__ import annotations

import numpy as np


def predict_dummy(patches: np.ndarray, coords: list[tuple[int, int]], seed: int = 42) -> np.ndarray:
    """
    Return fake anomaly scores for each patch.

    Uses a simple heuristic: mean intensity in a few bands + spatial variation,
    then maps to [0, 1] for a plausible-looking heatmap.

    Args:
        patches: (N, H, W, C) float32
        coords: list of (y, x) for each patch
        seed: for reproducibility

    Returns:
        scores: (N,) float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)
    n = len(patches)
    if n == 0:
        return np.array([], dtype=np.float32)

    # Heuristic: combine mean intensity (some bands) + local std + noise
    # This gives spatially varying "scores" that look like a heatmap
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p = patches[i]  # (H, W, C)
        # Mean over a few bands (simulate "suspicious" spectral signature)
        band_mean = np.mean(p[:, :, : min(4, p.shape[2])])
        # Spatial variance (texture)
        spatial_std = np.std(p)
        # Add coord-dependent noise for spatial structure
        y, x = coords[i]
        coord_factor = np.sin(y * 0.02) * np.cos(x * 0.02)
        raw = 0.3 * band_mean + 0.2 * spatial_std + 0.3 * coord_factor + 0.2 * rng.random()
        scores[i] = float(np.clip(raw, 0, 1))

    # Normalize to [0, 1] for better heatmap contrast
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores.astype(np.float32)
