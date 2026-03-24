#!/usr/bin/env python3
"""
Eksporter trent Baseline3DCNN (eller annen modell fra registry) til ONNX + manifest.json
for bruk i SpectralAssist med Onnx3DCnnClassifier (input NCDHW: 1,1,C,H,W).

Kjør etter valgt checkpoint (se docs/CNN3D_ONNX_WORKFLOW.md).

Eksempel:
  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export_cnn3d_onnx.py \\
    --checkpoint outputs/checkpoints/best.pt \\
    --model-config configs/models/baseline_3dcnn.yaml \\
    --spectral-bands 16 --patch-h 64 --patch-w 64 \\
    --out-dir outputs/onnx_cnn3d_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import build_model_from_config


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3D CNN to ONNX + manifest for C#.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model YAML (default: ckpt['model_config_path'] if present)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for .onnx + manifest")
    parser.add_argument("--spectral-bands", type=int, required=True, help="C in (1,1,C,H,W) — e.g. 16")
    parser.add_argument("--patch-h", type=int, required=True, help="Patch height (must match training)")
    parser.add_argument("--patch-w", type=int, required=True, help="Patch width (must match training)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--onnx-name", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ckpt = _load_checkpoint(ckpt_path, device)
    cfg_path = args.model_config
    if cfg_path is None:
        raw = ckpt.get("model_config_path")
        if raw is None:
            raise ValueError("Provide --model-config or store 'model_config_path' in checkpoint")
        cfg_path = Path(raw)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    model = build_model_from_config(cfg_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    c, h, w = args.spectral_bands, args.patch_h, args.patch_w
    dummy = torch.randn(1, 1, c, h, w, device=device, dtype=torch.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / args.onnx_name

    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    manifest = {
        "schema_version": "1.1",
        "generator": "export_cnn3d_onnx.py",
        "metadata": {
            "name": str(ckpt.get("run_name", "cnn3d_export")),
            "version": "1.0.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "author": "ML pipeline",
            "description": "3D CNN (NCDHW); matches pytorch_backend (N,H,W,C) -> (N,1,C,H,W)",
        },
        "training_config": {
            "checkpoint": str(ckpt_path),
            "model_config": str(cfg_path),
        },
        "input_spec": {
            "input_rank": 5,
            "tensor_layout": "NCDHW",
            "input_shape": [1, 1, c, h, w],
            "spectral_bands": c,
            "expected_bands": c,
            "spatial_patch_size": [h, w],
            "dtype": "float32",
        },
        "output_spec": {
            "type": "logits",
            "classes": ["class_0", "class_1"],
        },
        "artifacts": {
            "pipeline_onnx": args.onnx_name,
        },
    }
    if isinstance(ckpt.get("classes"), list):
        manifest["output_spec"]["classes"] = [str(x) for x in ckpt["classes"]]

    out_manifest = args.out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote ONNX: {onnx_path}")
    print(f"Wrote manifest: {out_manifest}")
    print(f"Static input shape (N,C,D,H,W) = (1, 1, {c}, {h}, {w})")


if __name__ == "__main__":
    main()
