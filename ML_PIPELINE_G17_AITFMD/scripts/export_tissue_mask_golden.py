#!/usr/bin/env python3
"""
Golden tensors for C# mean_std_percentile tissue mask parity tests.

Run from ML pipeline root with venv activated:

  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export_tissue_mask_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.tissue_mask import build_tissue_mask

GUI_ROOT = (
    PROJECT_ROOT.parent
    / "GUI_G17_AITFMD"
    / "spectral-assist"
    / "SpectralAssist.Tests"
    / "Fixtures"
    / "tissue_mask_golden"
)


def main() -> None:
    rng = np.random.default_rng(42)
    h, w, b = 16, 16, 16
    cube = rng.random((h, w, b), dtype=np.float32)
    q_mean, q_std = 0.5, 0.4
    min_object_size, min_hole_size = 4, 4

    mask = build_tissue_mask(
        cube,
        method="mean_std_percentile",
        min_object_size=min_object_size,
        min_hole_size=min_hole_size,
        q_mean=q_mean,
        q_std=q_std,
    )

    GUI_ROOT.mkdir(parents=True, exist_ok=True)
    flat = np.ascontiguousarray(cube, dtype=np.float32).ravel(order="C")
    GUI_ROOT.joinpath("cube_hwb_f32.bin").write_bytes(flat.tobytes())
    GUI_ROOT.joinpath("mask_uint8.bin").write_bytes(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    )

    mean = cube.mean(axis=2, dtype=np.float32)
    std = cube.std(axis=2, dtype=np.float32)
    GUI_ROOT.joinpath("mean_map_f32.bin").write_bytes(
        np.ascontiguousarray(mean, dtype=np.float32).tobytes()
    )
    GUI_ROOT.joinpath("std_map_f32.bin").write_bytes(
        np.ascontiguousarray(std, dtype=np.float32).tobytes()
    )

    manifest = {
        "layout": "hwb_c_order",
        "lines": h,
        "samples": w,
        "bands": b,
        "method": "mean_std_percentile",
        "q_mean": q_mean,
        "q_std": q_std,
        "min_object_size": min_object_size,
        "min_hole_size": min_hole_size,
        "t_mean": float(np.quantile(mean, q_mean)),
        "t_std": float(np.quantile(std, q_std)),
    }
    GUI_ROOT.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote golden files to: {GUI_ROOT}")


if __name__ == "__main__":
    main()
