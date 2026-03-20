"""On-the-fly patch sampling from cubes."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.patching import load_mask, load_numpy_cube


class CubePatchDataset(Dataset):
    """
    Dataset that loads cubes and samples patches on-the-fly.

    Uses cube-level manifest (output_path, label_id, split, patient_id, roi_name).
    Optionally filters by tissue mask (min_tissue_ratio).

    patches_per_cube: antall patches per kube per epoke. 1 = én tilfeldig.
        Høyere tall = flere ulike patches per kube. "all" = alle gyldige posisjoner
        (krever stride_h, stride_w).
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        patch_h: int = 64,
        patch_w: int = 64,
        mask_root: Path | None = None,
        min_tissue_ratio: float = 0.0,
        val_seed: int | None = None,
        cube_root: Path | None = None,
        patches_per_cube: int = 1,
        stride_h: int = 32,
        stride_w: int = 32,
        use_all_patches: bool = False,
    ):
        required = {"output_path", "label_id", "split"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Cube manifest missing columns: {sorted(missing)}")

        self.rows = rows.reset_index(drop=True)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.mask_root = Path(mask_root) if mask_root else None
        self.min_tissue_ratio = min_tissue_ratio
        self.val_seed = val_seed
        self._is_val = val_seed is not None
        self.cube_root = Path(cube_root) if cube_root else None
        self.patches_per_cube = max(1, int(patches_per_cube))
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.use_all_patches = use_all_patches

        if use_all_patches:
            self._all_positions: list[tuple[int, int, int]] = []  # (cube_idx, y, x)
            for cube_idx in range(len(self.rows)):
                cube_path = self._resolve_cube_path_static(cube_idx)
                if not cube_path.exists():
                    continue
                cube = load_numpy_cube(cube_path)
                mask = self._load_mask_for_row(self.rows.iloc[cube_idx])
                for y, x in self._valid_positions_static(cube, mask):
                    self._all_positions.append((cube_idx, y, x))

    def _resolve_cube_path_static(self, row_idx: int) -> Path:
        row = self.rows.iloc[row_idx]
        if self.cube_root and "patient_id" in row and "roi_name" in row:
            return self.cube_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        return Path(str(row["output_path"]))

    def _load_mask_for_row(self, row: pd.Series) -> np.ndarray | None:
        if not self.mask_root or "patient_id" not in row or "roi_name" not in row:
            return None
        mask_path = self.mask_root / str(row["patient_id"]) / f"{row['roi_name']}_mask.npy"
        if not mask_path.exists():
            return None
        try:
            return load_mask(mask_path)
        except Exception:
            return None

    def _valid_positions_static(
        self, cube: np.ndarray, mask: np.ndarray | None
    ) -> list[tuple[int, int]]:
        h, w, _ = cube.shape
        if h < self.patch_h or w < self.patch_w:
            return []
        positions: list[tuple[int, int]] = []
        use_mask = mask is not None and self.min_tissue_ratio > 0
        for y in range(0, h - self.patch_h + 1, self.stride_h):
            for x in range(0, w - self.patch_w + 1, self.stride_w):
                if use_mask:
                    patch_mask = mask[y : y + self.patch_h, x : x + self.patch_w]
                    if float(np.mean(patch_mask > 0)) < self.min_tissue_ratio:
                        continue
                positions.append((y, x))
        return positions

    def __len__(self) -> int:
        if self.use_all_patches:
            return len(self._all_positions)
        return len(self.rows) * self.patches_per_cube

    def _valid_positions(
        self,
        cube: np.ndarray,
        mask: np.ndarray | None,
    ) -> list[tuple[int, int]]:
        h, w, _ = cube.shape
        if h < self.patch_h or w < self.patch_w:
            return []

        positions: list[tuple[int, int]] = []
        use_mask = mask is not None and self.min_tissue_ratio > 0

        for y in range(0, h - self.patch_h + 1):
            for x in range(0, w - self.patch_w + 1):
                if use_mask:
                    patch_mask = mask[y : y + self.patch_h, x : x + self.patch_w]
                    if float(np.mean(patch_mask > 0)) < self.min_tissue_ratio:
                        continue
                positions.append((y, x))

        return positions

    def _resolve_cube_path(self, row: pd.Series) -> Path:
        if self.cube_root and "patient_id" in row and "roi_name" in row:
            return self.cube_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        return Path(str(row["output_path"]))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_all_patches:
            cube_idx, y, x = self._all_positions[idx]
            row = self.rows.iloc[cube_idx]
            cube_path = self._resolve_cube_path(row)
            cube = load_numpy_cube(cube_path)
            patch = cube[y : y + self.patch_h, x : x + self.patch_w, :]
        else:
            cube_idx = idx // self.patches_per_cube
            row = self.rows.iloc[cube_idx]
            cube_path = self._resolve_cube_path(row)
            if not cube_path.exists():
                raise FileNotFoundError(f"Cube not found: {cube_path}")
            cube = load_numpy_cube(cube_path)
            mask = None
            if self.mask_root and "patient_id" in row and "roi_name" in row:
                mask_path = self.mask_root / str(row["patient_id"]) / f"{row['roi_name']}_mask.npy"
                if mask_path.exists():
                    mask = load_mask(mask_path)
            positions = self._valid_positions(cube, mask)
            if not positions:
                h, w, _ = cube.shape
                y = max(0, min(h - self.patch_h, (h - self.patch_h) // 2))
                x = max(0, min(w - self.patch_w, (w - self.patch_w) // 2))
                y, x = int(y), int(x)
            else:
                if self._is_val and self.val_seed is not None:
                    rng = random.Random(self.val_seed + idx)
                    y, x = rng.choice(positions)
                else:
                    y, x = random.choice(positions)
            patch = cube[y : y + self.patch_h, x : x + self.patch_w, :]

        x_t = torch.from_numpy(patch.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        y_t = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x_t, y_t
