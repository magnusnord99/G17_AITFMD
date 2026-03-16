"""
Build heatmap from patch-level predictions.

Aggregates overlapping patch scores into a smooth (H, W) heatmap.
"""

from __future__ import annotations

import numpy as np


def build_heatmap(
    coords: list[tuple[int, int]],
    scores: np.ndarray,
    patch_h: int,
    patch_w: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Build heatmap by averaging patch scores over overlapping regions.

    For each patch at (y, x) with score s, add s to heatmap[y:y+patch_h, x:x+patch_w]
    and increment a count. Final heatmap = sum / count.

    Args:
        coords: list of (y, x) top-left for each patch
        scores: (N,) anomaly score per patch
        patch_h, patch_w: patch dimensions
        out_h, out_w: output heatmap dimensions (e.g. cube shape)

    Returns:
        heatmap: (out_h, out_w) float32 in [0, 1]
    """
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    count = np.zeros((out_h, out_w), dtype=np.float32)

    for (y, x), score in zip(coords, scores):
        y_end = min(y + patch_h, out_h)
        x_end = min(x + patch_w, out_w)
        y_slice = slice(y, y_end)
        x_slice = slice(x, x_end)
        heatmap[y_slice, x_slice] += score
        count[y_slice, x_slice] += 1

    # Avoid div by zero
    count = np.maximum(count, 1e-6)
    heatmap = heatmap / count

    # Clip to [0, 1]
    return np.clip(heatmap, 0, 1).astype(np.float32)
