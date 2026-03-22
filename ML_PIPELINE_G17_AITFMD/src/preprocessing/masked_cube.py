"""Vevsmaske på HSI-kuber (H, W, C) — felles for build- og fit-skript."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mask_path_for_roi(mask_root: Path, patient_id: str, roi_name: str) -> Path:
    """Standardsti: mask_root / patient_id / {roi_name}_mask.npy"""
    return Path(mask_root) / str(patient_id) / f"{roi_name}_mask.npy"


def load_binary_mask(path: Path) -> np.ndarray:
    """Last maske som (H, W) float32 i {0, 1}."""
    m = np.load(path, allow_pickle=False)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D (H,W), got {m.shape} at {path}")
    m = m.astype(np.float32, copy=False)
    if float(m.max()) > 1.0:
        m = (m > 0).astype(np.float32)
    return m


def apply_mask_to_cube(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Multipliser bakgrunn til 0: cube (H,W,C) * mask (H,W).

    Returnerer ny array (float32-kompatibel kopi ved behov).
    """
    if mask.shape[:2] != cube.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} must match cube spatial {cube.shape[:2]}"
        )
    m = mask.astype(np.float32, copy=False)
    if float(m.max()) > 1.0:
        m = (m > 0).astype(np.float32)
    cube_f = np.asarray(cube, dtype=np.float32)
    return cube_f * m[..., np.newaxis]


def maybe_apply_mask_from_disk(
    cube: np.ndarray,
    mask_root: Path | None,
    patient_id: str,
    roi_name: str,
    *,
    require_mask: bool,
) -> np.ndarray:
    """
    Hvis mask_root er satt: last maske og apply_mask_to_cube.
    Manglende fil: FileNotFoundError hvis require_mask, ellers uendret cube.
    """
    cube2, _ = prepare_cube_with_mask(
        cube,
        mask_root,
        patient_id,
        roi_name,
        require_mask=require_mask,
        apply_to_cube=True,
    )
    return cube2


def prepare_cube_with_mask(
    cube: np.ndarray,
    mask_root: Path | None,
    patient_id: str,
    roi_name: str,
    *,
    require_mask: bool,
    apply_to_cube: bool,
) -> tuple[np.ndarray, bool]:
    """
    Last maske fra disk hvis mask_root er satt.

    Returns:
        (cube_out, had_mask): had_mask True hvis maske fantes og ble brukt.
    """
    if mask_root is None:
        return cube, False
    mp = mask_path_for_roi(mask_root, patient_id, roi_name)
    if not mp.exists():
        if require_mask:
            raise FileNotFoundError(
                f"require_mask=True but mask not found: {mp}"
            )
        return cube, False
    if not apply_to_cube:
        return cube, True
    mask = load_binary_mask(mp)
    return apply_mask_to_cube(cube, mask), True
