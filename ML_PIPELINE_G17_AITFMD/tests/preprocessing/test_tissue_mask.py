"""Unit tests for tissue mask utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.tissue_mask import build_tissue_mask, save_tissue_mask, tissue_ratio


class TestTissueMask(unittest.TestCase):
    def test_build_mean_otsu_picks_darker_region_as_tissue(self) -> None:
        cube = np.ones((20, 20, 6), dtype=np.float32) * 0.9
        cube[4:16, 5:15, :] = 0.2

        mask = build_tissue_mask(
            cube,
            method="mean_otsu",
            min_object_size=4,
            min_hole_size=4,
        )

        self.assertEqual(mask.shape, (20, 20))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertGreater(mask[10, 10], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_unsupported_method_raises(self) -> None:
        cube = np.ones((10, 10, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_tissue_mask(cube, method="invalid")

    def test_tissue_ratio_computes_fraction(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        self.assertAlmostEqual(tissue_ratio(mask), 0.25)

    def test_save_mask_writes_npy(self) -> None:
        mask = np.array([[True, False], [False, True]])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mask.npy"
            saved = save_tissue_mask(mask, out_path)
            loaded = np.load(saved)
            np.testing.assert_array_equal(loaded, np.array([[1, 0], [0, 1]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
