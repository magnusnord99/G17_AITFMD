#!/usr/bin/env python3
"""
Golden tensors for C# wavelet parity tests.

Run from ML pipeline root with venv activated:

  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export_wavelet_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.wavelet import reduce_cube_wavelet_approx_padded

GUI_ROOT = (
    PROJECT_ROOT.parent
    / "GUI_G17_AITFMD"
    / "spectral-assist"
    / "SpectralAssist.Tests"
    / "Fixtures"
    / "wavelet_golden"
)


def main() -> None:
    rng = np.random.default_rng(44)
    h, w, b = 4, 4, 275
    target_bands = 16
    wavelet = "db2"
    mode = "periodization"
    pad_mode = "edge"

    cube = rng.random((h, w, b), dtype=np.float32)
    reduced = reduce_cube_wavelet_approx_padded(
        cube,
        target_bands=target_bands,
        wavelet=wavelet,
        level=None,
        mode=mode,
        pad_mode=pad_mode,
    )

    GUI_ROOT.mkdir(parents=True, exist_ok=True)
    GUI_ROOT.joinpath("cube_hwb_f32.bin").write_bytes(
        np.ascontiguousarray(cube, dtype=np.float32).ravel(order="C").tobytes()
    )
    GUI_ROOT.joinpath("wavelet16_expect.bin").write_bytes(
        np.ascontiguousarray(reduced, dtype=np.float32).ravel(order="C").tobytes()
    )

    manifest = {
        "layout": "hwb_c_order",
        "lines": h,
        "samples": w,
        "bands_in": b,
        "bands_out": target_bands,
        "feature_mode": "approx_padded",
        "wavelet": wavelet,
        "mode": mode,
        "pad_mode": pad_mode,
    }
    GUI_ROOT.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote golden files to: {GUI_ROOT}")


if __name__ == "__main__":
    main()
