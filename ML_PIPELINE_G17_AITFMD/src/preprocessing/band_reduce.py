"""Simple band reduction by averaging — baseline for comparison with PCA/AE/wavelet."""

from __future__ import annotations

import numpy as np


def _compute_bin_sizes(n_in: int, n_out: int, strategy: str) -> list[int]:
    """
    Compute how many input bands go into each output band.

    Strategies:
    - "crop": Crop n_in to largest multiple of n_out, then divide evenly.
      E.g. 275 -> crop to 272 -> 16 groups of 17.
    - "uneven": Distribute remainder across first groups. No info loss.
      E.g. 275 -> 13 groups of 17 + 3 groups of 18.
    """
    if strategy == "crop":
        n_usable = (n_in // n_out) * n_out
        if n_usable == 0:
            raise ValueError(f"Cannot crop {n_in} bands to {n_out} — too few bands.")
        base = n_usable // n_out
        return [base] * n_out
    if strategy == "uneven":
        base, remainder = divmod(n_in, n_out)
        # First `remainder` groups get base+1, rest get base
        return [base + 1] * remainder + [base] * (n_out - remainder)
    raise ValueError(f"Unknown strategy '{strategy}'. Use 'crop' or 'uneven'.")


def reduce_bands_by_avg(
    cube: np.ndarray,
    n_out_bands: int,
    strategy: str = "crop",
) -> np.ndarray:
    """
    Reduce spectral bands by averaging adjacent bands.

    Args:
        cube: (H, W, B) hyperspectral cube.
        n_out_bands: Number of output bands.
        strategy: "crop" (crop to divisible, even groups) or "uneven" (no crop, variable group sizes).

    Returns:
        (H, W, n_out_bands) reduced cube.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (H, W, B), got shape={cube.shape}")
    h, w, n_in = cube.shape

    bin_sizes = _compute_bin_sizes(n_in, n_out_bands, strategy)
    out = np.empty((h, w, n_out_bands), dtype=np.float32)
    start = 0
    for i, size in enumerate(bin_sizes):
        chunk = cube[:, :, start : start + size]
        out[:, :, i] = np.mean(chunk, axis=2, dtype=np.float32)
        start += size
    return out
