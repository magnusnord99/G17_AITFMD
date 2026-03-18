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
    Dataset that loads cubes and samples a random patch per __getitem__.

    Uses cube-level manifest (output_path, label_id, split, patient_id, roi_name).
    Optionally filters by tissue mask (min_tissue_ratio).
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

    def __len__(self) -> int:
        return len(self.rows)

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
        row = self.rows.iloc[idx]
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
            # Fallback: use center patch if no valid positions
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
        # (H, W, D) -> (1, D, H, W) for Conv3d
        x_t = torch.from_numpy(patch.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        y_t = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x_t, y_t
