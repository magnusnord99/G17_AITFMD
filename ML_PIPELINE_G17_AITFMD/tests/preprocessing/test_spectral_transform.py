"""Unit tests for spectral transform helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.spectral_transform import reduce_bands_neighbor_average


class TestSpectralTransform(unittest.TestCase):
    def test_window_one_returns_input_values(self) -> None:
        cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        reduced = reduce_bands_neighbor_average(cube, window=1)
        np.testing.assert_array_equal(reduced, cube)

    def test_reduction_uses_neighbor_mean_and_trims_tail(self) -> None:
        cube = np.arange(2 * 2 * 7, dtype=np.float32).reshape(2, 2, 7)
        reduced = reduce_bands_neighbor_average(cube, window=3)

        expected = np.stack(
            [
                cube[:, :, 0:3].mean(axis=2),
                cube[:, :, 3:6].mean(axis=2),
            ],
            axis=2,
        ).astype(np.float32)

        self.assertEqual(reduced.shape, (2, 2, 2))
        self.assertEqual(reduced.dtype, np.float32)
        np.testing.assert_allclose(reduced, expected, rtol=1e-6, atol=1e-6)

    def test_window_too_large_raises(self) -> None:
        cube = np.ones((2, 2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            reduce_bands_neighbor_average(cube, window=8)


if __name__ == "__main__":
    unittest.main()
