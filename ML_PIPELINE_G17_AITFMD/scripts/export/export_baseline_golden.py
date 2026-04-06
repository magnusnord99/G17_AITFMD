#!/usr/bin/env python3
"""
Write float32 golden tensors for C# preprocessing parity tests.

Activate the ML venv first (same as training), then run from the ML pipeline root:

  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export_baseline_golden.py

Requires: numpy, and imports from src.preprocessing (same math as training pipeline).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.band_reduce import reduce_bands_by_avg
from src.preprocessing.spectral_transform import reduce_bands_neighbor_average


def calibrate_cube(raw: np.ndarray, dark: np.ndarray, white: np.ndarray, eps: float) -> np.ndarray:
    """Same as calibrateClip.calibrate_cube (no spectral dependency for this script)."""
    return ((raw - dark) / (white - dark + eps)).astype(np.float32)


def clip_cube(cube: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    return np.clip(cube, clip_min, clip_max)

# Output next to C# test project
GUI_ROOT = PROJECT_ROOT.parent / "GUI_G17_AITFMD" / "spectral-assist" / "SpectralAssist.Tests" / "Fixtures" / "baseline_golden"


def _write_f32_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel(order="C")
    path.write_bytes(flat.tobytes())


def _write_manifest(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    eps = 1e-8
    clip_min, clip_max = 0.0, 1.0
    window = 3

    GUI_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Small chain: 4x4x9 -> avg3 -> 4x4x3 ---
    rng = np.random.default_rng(42)
    raw = (rng.random((4, 4, 9), dtype=np.float32) * 0.5).astype(np.float32)
    dark = np.zeros((4, 4, 9), dtype=np.float32)
    white = np.ones((4, 4, 9), dtype=np.float32)

    cal = calibrate_cube(raw, dark, white, eps)
    clipped = clip_cube(cal, clip_min, clip_max)
    avg3 = reduce_bands_neighbor_average(clipped, window)

    _write_f32_bin(GUI_ROOT / "small_raw.bin", raw)
    _write_f32_bin(GUI_ROOT / "small_dark.bin", dark)
    _write_f32_bin(GUI_ROOT / "small_white.bin", white)
    _write_f32_bin(GUI_ROOT / "small_expect_after_avg3.bin", avg3)

    # --- Full spectral reduction: 4x4x275 -> avg3 -> 91 -> reduce to 16 (crop) ---
    rng2 = np.random.default_rng(43)
    raw275 = (rng2.random((4, 4, 275), dtype=np.float32) * 0.5).astype(np.float32)
    dark275 = np.zeros((4, 4, 275), dtype=np.float32)
    white275 = np.ones((4, 4, 275), dtype=np.float32)

    cal275 = calibrate_cube(raw275, dark275, white275, eps)
    clip275 = clip_cube(cal275, clip_min, clip_max)
    avg3_275 = reduce_bands_neighbor_average(clip275, window)
    out16 = reduce_bands_by_avg(avg3_275, n_out_bands=16, strategy="crop")

    _write_f32_bin(GUI_ROOT / "chain275_raw.bin", raw275)
    _write_f32_bin(GUI_ROOT / "chain275_dark.bin", dark275)
    _write_f32_bin(GUI_ROOT / "chain275_white.bin", white275)
    _write_f32_bin(GUI_ROOT / "chain275_expect_avg16.bin", out16)

    manifest = {
        "layout": "hwb_c_order",
        "dtype": "float32",
        "epsilon": eps,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "neighbor_window": window,
        "small": {
            "lines": 4,
            "samples": 4,
            "bands_in": 9,
            "bands_after_avg3": 3,
        },
        "chain275": {
            "lines": 4,
            "samples": 4,
            "bands_in": 275,
            "bands_after_avg3": int(avg3_275.shape[2]),
            "bands_out": 16,
            "band_reduce_strategy": "crop",
        },
    }
    _write_manifest(manifest, GUI_ROOT / "manifest.json")

    print(f"Wrote golden files to: {GUI_ROOT}")


if __name__ == "__main__":
    main()
