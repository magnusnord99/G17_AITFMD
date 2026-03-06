"""PCA utilities for hyperspectral cubes."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def flatten_cube(cube: np.ndarray) -> np.ndarray:
    """Flatten cube from (H, W, B) to (H*W, B)."""
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube (H, W, B), got shape={cube.shape}")
    h, w, b = cube.shape
    return cube.reshape(h * w, b)


def restore_cube(flat_features: np.ndarray, h: int, w: int) -> np.ndarray:
    """Restore flattened features from (H*W, C) back to (H, W, C)."""
    if flat_features.ndim != 2:
        raise ValueError(
            f"Expected 2D features (H*W, C), got shape={flat_features.shape}"
        )
    if flat_features.shape[0] != h * w:
        raise ValueError(
            f"Cannot reshape: features rows={flat_features.shape[0]} but h*w={h*w}"
        )
    c = flat_features.shape[1]
    return flat_features.reshape(h, w, c).astype(np.float32)


def fit_pca_from_pixels(
    train_pixels: np.ndarray,
    n_components: int,
    random_state: int = 42,
    svd_solver: str = "auto",
) -> PCA:
    """
    Fit PCA on train pixels only.

    train_pixels must be shape (N, B), where N is number of sampled train pixels.
    """
    if train_pixels.ndim != 2:
        raise ValueError(
            f"Expected train_pixels with shape (N, B), got {train_pixels.shape}"
        )
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    pca.fit(train_pixels)
    return pca


def transform_cube_with_pca(cube: np.ndarray, pca_model: PCA) -> np.ndarray:
    """Apply a fitted PCA model to one cube and return shape (H, W, C)."""
    h, w, _ = cube.shape
    flat = flatten_cube(cube)
    transformed = pca_model.transform(flat)
    return restore_cube(transformed, h, w)

