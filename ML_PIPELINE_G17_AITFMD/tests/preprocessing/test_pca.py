"""Unit tests for PCA preprocessing utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.preprocessing.pca import (
        fit_pca_from_pixels,
        flatten_cube,
        restore_cube,
        transform_cube_with_pca,
    )
except ModuleNotFoundError as exc:
    PCA_IMPORT_ERROR = exc
    PCA_AVAILABLE = False
else:
    PCA_IMPORT_ERROR = None
    PCA_AVAILABLE = True


@unittest.skipUnless(
    PCA_AVAILABLE,
    f"PCA module unavailable in this environment ({PCA_IMPORT_ERROR})",
)
class TestPCAUtils(unittest.TestCase):
    def test_flatten_and_restore_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        cube = rng.normal(size=(6, 5, 8)).astype(np.float32)

        flat = flatten_cube(cube)
        self.assertEqual(flat.shape, (30, 8))

        restored = restore_cube(flat, 6, 5)
        self.assertEqual(restored.shape, cube.shape)
        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_allclose(restored, cube, rtol=1e-6, atol=1e-6)

    def test_flatten_rejects_non_3d(self) -> None:
        with self.assertRaises(ValueError):
            flatten_cube(np.ones((4, 5), dtype=np.float32))

    def test_restore_rejects_wrong_rows(self) -> None:
        flat = np.ones((11, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            restore_cube(flat, h=3, w=4)

    def test_fit_rejects_non_2d_pixels(self) -> None:
        with self.assertRaises(ValueError):
            fit_pca_from_pixels(np.ones((2, 3, 4), dtype=np.float32), n_components=2)

    def test_fit_and_transform_cube(self) -> None:
        rng = np.random.default_rng(0)
        train_pixels = rng.normal(size=(200, 12)).astype(np.float32)
        cube = rng.normal(size=(10, 7, 12)).astype(np.float32)

        pca = fit_pca_from_pixels(train_pixels, n_components=5, random_state=42)
        transformed = transform_cube_with_pca(cube, pca)

        self.assertEqual(transformed.shape, (10, 7, 5))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
