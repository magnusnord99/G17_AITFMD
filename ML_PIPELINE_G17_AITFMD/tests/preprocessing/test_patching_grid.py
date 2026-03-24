"""Tests for patch grid counting vs iter_patches."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.patching import count_patch_grid, iter_patches


class TestPatchGridCount(unittest.TestCase):
    def test_count_matches_iter_patches_no_mask(self) -> None:
        cube = np.zeros((64, 64, 4), dtype=np.float32)
        stats = count_patch_grid(cube, 32, 32, 16, 16, mask=None, min_tissue_ratio=0.6)
        coords = list(iter_patches(cube, 32, 32, 16, 16, mask=None, min_tissue_ratio=0.6))
        self.assertEqual(stats["evaluated"], len(coords))
        self.assertEqual(stats["total_possible"], stats["evaluated"])
        self.assertEqual(stats["filtered_by_tissue"], 0)

    def test_count_matches_iter_patches_with_tissue_filter(self) -> None:
        cube = np.zeros((64, 64, 2), dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :32] = 1  # only top-left quadrant has tissue
        stats = count_patch_grid(cube, 32, 32, 32, 32, mask=mask, min_tissue_ratio=0.5)
        coords = list(iter_patches(cube, 32, 32, 32, 32, mask=mask, min_tissue_ratio=0.5))
        self.assertEqual(stats["evaluated"], len(coords))
        self.assertEqual(stats["total_possible"], 4)
        self.assertEqual(stats["filtered_by_tissue"], 3)
        self.assertEqual(stats["evaluated"], 1)

    def test_too_small_cube_returns_zeros(self) -> None:
        cube = np.zeros((10, 10, 2), dtype=np.float32)
        stats = count_patch_grid(cube, 32, 32, 16, 16, mask=None, min_tissue_ratio=0.0)
        self.assertEqual(stats, {"total_possible": 0, "filtered_by_tissue": 0, "evaluated": 0})


if __name__ == "__main__":
    unittest.main()
