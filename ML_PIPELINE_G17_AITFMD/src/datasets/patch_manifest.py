"""Datasets for patch-based 3D CNN training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatchManifestDataset(Dataset):
    """Dataset that reads patch paths from manifest.csv."""

    def __init__(self, rows: pd.DataFrame):
        required = {"patch_path", "label_id"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Manifest rows missing columns: {sorted(missing)}")
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        patch_path = Path(str(row["patch_path"]))
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch not found: {patch_path}")

        arr = np.load(patch_path, allow_pickle=False).astype(np.float32, copy=False)

        # Expected from preprocessing: (H, W, D). Convert to (C, D, H, W) with C=1.
        if arr.ndim == 3:
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            x = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported patch shape {arr.shape} at {patch_path}")

        y = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return x, y
