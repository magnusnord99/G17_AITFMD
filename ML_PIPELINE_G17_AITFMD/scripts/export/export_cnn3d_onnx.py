#!/usr/bin/env python3
"""
Eksporter trent 3D-CNN til ONNX + manifest.json for SpectralAssist (NCDHW: 1,1,C,H,W).

manifest.json: preprocessing hentes fra configs/preprocessing/pipeline.yaml (calibration, clip, avg3, tissue_mask).
spectral_reduction og patching i den YAML-en brukes ikke her — bruk --spectral-bands / --reducer-method. Valgfri JSON-mal: --manifest-template.

Kjør etter valgt checkpoint (se docs/CNN3D_ONNX_WORKFLOW.md).

Eksempel:
  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export/export_cnn3d_onnx.py \\
    --checkpoint outputs/checkpoints/best.pt \\
    --model-config configs/models/baseline_3dcnn.yaml \\
    --spectral-bands 16 --patch-h 64 --patch-w 64 \\
    --out-dir outputs/onnx_cnn3d_v1 \\
    --dataset wavelet_manifest.csv \\
    --reducer-method wavelet
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import build_model_from_config

DEFAULT_PIPELINE_YAML = PROJECT_ROOT / "configs" / "preprocessing" / "pipeline.yaml"


def _step_enabled(section: dict[str, Any] | None) -> bool:
    if not section:
        return False
    return bool(section.get("enabled", True))


def preprocessing_from_pipeline_yaml(path: Path) -> dict[str, Any]:
    """
    Bygg manifest-preprocessing fra ML preprocessing YAML (pipeline.calibration … tissue_mask).
    Hopper over spectral_reduction og patching — de speiles i input_spec / spectral_reducer ved eksport.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    p = raw.get("pipeline") or {}
    steps: list[str] = []
    params: dict[str, Any] = {
        "band_reduce_out_bands": 0,
        "band_reduce_strategy": "",
    }

    if _step_enabled(p.get("calibration")):
        steps.append("calibrate")
        c = p["calibration"]
        params["calibration_epsilon"] = float(c.get("eps", 1e-8))

    if _step_enabled(p.get("clip")):
        steps.append("clip")
        c = p["clip"]
        params["clip_min"] = float(c.get("clip_min", 0.0))
        params["clip_max"] = float(c.get("clip_max", 1.0))

    if _step_enabled(p.get("avg3")):
        steps.append("neighbor_average")
        a = p["avg3"]
        params["neighbor_average_window"] = int(a.get("reduction_window", 3))

    if _step_enabled(p.get("tissue_mask")):
        steps.append("tissue_mask")
        t = p["tissue_mask"]
        params["tissue_mask_method"] = str(t.get("method", "mean_std_percentile"))
        params["tissue_mask_q_mean"] = float(t.get("q_mean", 0.5))
        params["tissue_mask_q_std"] = float(t.get("q_std", 0.4))
        params["tissue_mask_min_object_size"] = int(t.get("min_object_size", 1000))
        params["tissue_mask_min_hole_size"] = int(t.get("min_hole_size", 1000))
        if "tissue_side" in t:
            params["tissue_mask_tissue_side"] = str(t["tissue_side"])

    return {"steps": steps, "params": params}


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'")
    return ckpt


def _layer_sequence_ordered(model: torch.nn.Module) -> list[str]:
    """Bladlag i dybde-første rekkefølge (typisk samme som forward)."""
    seq: list[str] = []
    for name, m in model.named_modules():
        if not name:
            continue
        if len(list(m.children())) > 0:
            continue
        seq.append(m.__class__.__name__)
    return seq


def _load_model_yaml(cfg_path: Path) -> dict[str, Any]:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _build_gui_manifest(
    *,
    base: dict[str, Any] | None,
    preprocessing: dict[str, Any],
    model: torch.nn.Module,
    model_yaml: dict[str, Any],
    cfg_path: Path,
    ckpt: dict[str, Any],
    spectral_bands: int,
    patch_h: int,
    patch_w: int,
    onnx_name: str,
    description: str | None,
    dataset: str | None,
    train_samples: int | None,
    reducer_method: str,
    embedded_reducer_in_onnx: bool,
) -> dict[str, Any]:
    """Bygg ferdig manifest. `preprocessing` kommer typisk fra preprocessing_from_pipeline_yaml."""
    arch = model_yaml.get("architecture") or {}
    model_block = model_yaml.get("model") or {}
    arch_name = str(model_block.get("name") or cfg_path.stem)
    num_classes = int(arch.get("num_classes", 2))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if isinstance(ckpt.get("classes"), list) and ckpt["classes"]:
        class_names = [str(x) for x in ckpt["classes"]]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    desc = description or (
        f"3D CNN ONNX export ({arch_name}); NCDHW shape [1,1,{spectral_bands},{patch_h},{patch_w}]"
    )

    if base is not None:
        out = copy.deepcopy(base)
        pipe = out.setdefault("pipeline", {})
        if "preprocessing" not in pipe:
            pipe["preprocessing"] = copy.deepcopy(preprocessing)
    else:
        out = {}
        out["pipeline"] = {"preprocessing": copy.deepcopy(preprocessing)}
        pipe = out["pipeline"]

    meta = out.setdefault("metadata", {})
    meta["name"] = arch_name
    meta["version"] = meta.get("version", "1.0.0")
    meta["created"] = created
    meta["author"] = meta.get("author", "ML Pipeline")
    meta["description"] = desc

    pipe["spectral_reducer"] = {
        "method": reducer_method,
        "embedded_in_onnx": embedded_reducer_in_onnx,
        "input_bands": spectral_bands,
        "output_bands": spectral_bands,
    }
    pipe["model"] = {
        "architecture": arch_name,
        "task": "classification",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layers": _layer_sequence_ordered(model),
    }

    out["input_spec"] = {
        "input_rank": 5,
        "tensor_layout": "NCDHW",
        "input_shape": [1, 1, spectral_bands, patch_h, patch_w],
        "spectral_bands": spectral_bands,
        "spatial_patch_size": [patch_h, patch_w],
        "dtype": "float32",
    }
    out["output_spec"] = {
        "type": "logits",
        "num_classes": num_classes,
        "classes": class_names[:num_classes] if len(class_names) >= num_classes else class_names,
    }
    out["training"] = {
        "dataset": dataset,
        "samples": train_samples,
        "epochs": ckpt.get("epoch"),
        "metrics": {"accuracy": None, "precision": None, "recall": None, "f1": None},
    }
    out["artifacts"] = {"model_onnx": onnx_name}
    out["validation"] = {"status": "pending", "result": None}
    out["schema_version"] = out.get("schema_version", "1.0")

    return out


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
    parser.add_argument(
        "--manifest-template",
        type=Path,
        default=None,
        help="Valgfri JSON som startpunkt (deepcopy); felt overskrives. Mangler preprocessing → fra --pipeline-config.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_YAML,
        help="preprocessing.pipeline i ML (calibration, clip, avg3, tissue_mask) → manifest preprocessing (default: configs/preprocessing/pipeline.yaml)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="metadata.description (standard: auto)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="training.dataset → null hvis utelatt")
    parser.add_argument("--train-samples", type=int, default=None, help="training.samples → null hvis utelatt")
    parser.add_argument(
        "--reducer-method",
        type=str,
        default="none",
        help="pipeline.spectral_reducer.method (none, wavelet, pca, …)",
    )
    parser.add_argument(
        "--embedded-reducer-in-onnx",
        action="store_true",
        help="pipeline.spectral_reducer.embedded_in_onnx true",
    )
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

    model_yaml = _load_model_yaml(cfg_path)
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

    base: dict[str, Any] | None = None
    if args.manifest_template is not None:
        tpl_path = args.manifest_template.resolve()
        if not tpl_path.is_file():
            raise FileNotFoundError(f"Manifest-mal finnes ikke: {tpl_path}")
        base = json.loads(tpl_path.read_text(encoding="utf-8"))

    pl_path = args.pipeline_config
    if not pl_path.is_absolute():
        pl_path = (PROJECT_ROOT / pl_path).resolve()
    if not pl_path.is_file():
        raise FileNotFoundError(f"Pipeline-config finnes ikke: {pl_path}")
    preprocessing_block = preprocessing_from_pipeline_yaml(pl_path)

    manifest = _build_gui_manifest(
        base=base,
        preprocessing=preprocessing_block,
        model=model,
        model_yaml=model_yaml,
        cfg_path=cfg_path,
        ckpt=ckpt,
        spectral_bands=c,
        patch_h=h,
        patch_w=w,
        onnx_name=args.onnx_name,
        description=args.description,
        dataset=args.dataset,
        train_samples=args.train_samples,
        reducer_method=args.reducer_method,
        embedded_reducer_in_onnx=args.embedded_reducer_in_onnx,
    )

    out_manifest = args.out_dir / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote ONNX: {onnx_path}")
    print(f"Wrote manifest: {out_manifest}")
    print(f"Static input shape (N,C,D,H,W) = (1, 1, {c}, {h}, {w})")


if __name__ == "__main__":
    main()
