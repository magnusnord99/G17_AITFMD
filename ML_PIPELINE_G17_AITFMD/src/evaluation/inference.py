"""Shared inference engine for ROI-level and patch-level evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class InferenceResult:
    """Container for the output of a single inference run.

    Attributes:
        probs: Probability for class 1, shape ``(N,)``.
        preds: Hard predicted class (0 or 1), shape ``(N,)``.
        labels: Ground-truth labels if available, shape ``(N,)`` or empty.
        roi_id: Identifier string for the ROI (``patient_id/roi_name``).
        n_patches: Number of patches evaluated.
    """

    probs: np.ndarray
    preds: np.ndarray
    labels: np.ndarray
    roi_id: str = ""
    n_patches: int = 0

    def __post_init__(self) -> None:
        self.n_patches = len(self.probs)


class InferenceEngine:
    """Batched patch inference over a 3-D hyperspectral cube.

    The engine is shared between ROI-level eval (``run_eval.py``) and
    patch-level eval (``run_eval_patch.py``) so the forward-pass logic is
    never duplicated.

    Args:
        model: Trained ``nn.Module`` already moved to ``device``.
        device: Inference device.
        batch_size: Number of patches per forward pass.
        amp: Whether to run inference under ``autocast``.  Should normally be
             ``False`` for eval (no memory benefit, avoids FP16 artifacts in probs).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 16,
        amp: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.amp = amp

    @torch.no_grad()
    def run_on_patches(
        self,
        patches: list[np.ndarray],
        labels: list[int] | None = None,
        roi_id: str = "",
    ) -> InferenceResult:
        """Run inference on a list of raw numpy patches.

        Args:
            patches: List of ``(H, W, C)`` float32 arrays.
            labels: Optional per-patch ground-truth labels (all same value for
                    ROI-level eval; needed for patch-level eval).
            roi_id: Human-readable identifier used in log messages.

        Returns:
            :class:`InferenceResult` with ``probs``, ``preds``, ``labels``.
        """
        if not patches:
            empty = np.array([], dtype=np.float32)
            return InferenceResult(
                probs=empty,
                preds=empty.astype(np.int64),
                labels=np.array(labels or [], dtype=np.int64),
                roi_id=roi_id,
            )

        all_probs: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        for i in range(0, len(patches), self.batch_size):
            chunk = patches[i : i + self.batch_size]
            # (H,W,C) -> (1,C,H,W); stack along dim 0 -> (B,1,C,H,W)
            tensors = [
                torch.from_numpy(p.astype(np.float32, copy=False))
                .permute(2, 0, 1)
                .unsqueeze(0)
                for p in chunk
            ]
            batch = torch.stack(tensors, dim=0).to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(batch)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)

        result_probs = np.concatenate(all_probs).astype(np.float32)
        result_preds = np.concatenate(all_preds).astype(np.int64)
        result_labels = np.asarray(labels, dtype=np.int64) if labels is not None else np.array([], dtype=np.int64)

        return InferenceResult(
            probs=result_probs,
            preds=result_preds,
            labels=result_labels,
            roi_id=roi_id,
        )

    def run_on_roi(
        self,
        cube_path: Path,
        patch_cfg: dict[str, Any],
        label: int | None = None,
        mask_path: Path | None = None,
        roi_id: str = "",
    ) -> InferenceResult:
        """Load a cube, extract its patch grid, and run inference.

        Args:
            cube_path: Path to the ``.npy`` HSI cube file.
            patch_cfg: Dict with keys ``patch_h``, ``patch_w``, ``stride_h``,
                       ``stride_w``, ``min_tissue_ratio``.
            label: Ground-truth label for the ROI (assigned to all patches).
            mask_path: Optional tissue mask ``.npy`` path.
            roi_id: Human-readable identifier.

        Returns:
            :class:`InferenceResult` for this ROI.
        """
        from src.preprocessing.patching import iter_patches, load_mask, load_numpy_cube

        cube = load_numpy_cube(cube_path)
        mask: np.ndarray | None = None
        if mask_path is not None and mask_path.exists():
            try:
                mask = load_mask(mask_path)
            except Exception as exc:
                log.warning("Could not load mask %s: %s", mask_path, exc)

        patches: list[np.ndarray] = []
        for patch, _y, _x in iter_patches(
            cube,
            patch_h=int(patch_cfg["patch_h"]),
            patch_w=int(patch_cfg["patch_w"]),
            stride_h=int(patch_cfg["stride_h"]),
            stride_w=int(patch_cfg["stride_w"]),
            mask=mask,
            min_tissue_ratio=float(patch_cfg.get("min_tissue_ratio", 0.0)),
        ):
            patches.append(patch)

        labels = [label] * len(patches) if label is not None else None
        result = self.run_on_patches(patches, labels=labels, roi_id=roi_id)
        log.debug("ROI %s: %d patches", roi_id, result.n_patches)
        return result
