#!/usr/bin/env python3
"""
Test av export_cnn3d_onnx: dummy checkpoint + sjekk at preprocessing matcher configs/preprocessing/pipeline.yaml.

Kjør alltid med prosjektets venv (PyTorch/ONNX-avhengigheter):

  cd ML_PIPELINE_G17_AITFMD
  source .venv/bin/activate
  python scripts/export/test_export_cnn3d_onnx.py

Kun manifest (ingen ONNX — raskt, trenger ikke onnx/onnxscript):

  python scripts/export/test_export_cnn3d_onnx.py --manifest-only

Full eksport (ONNX + manifest) til outputs/_test_onnx_export/:

  python scripts/export/test_export_cnn3d_onnx.py --full
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model_from_config

EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export" / "export_cnn3d_onnx.py"
DEFAULT_MODEL_YAML = PROJECT_ROOT / "configs" / "models" / "baseline_3dcnn.yaml"
DUMMY_CKPT = PROJECT_ROOT / "outputs" / "_test_export" / "dummy_baseline.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "_test_onnx_export"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_cnn3d_onnx", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Kunne ikke laste {EXPORT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dummy_checkpoint(model_yaml: Path) -> Path:
    DUMMY_CKPT.parent.mkdir(parents=True, exist_ok=True)
    rel = str(model_yaml.relative_to(PROJECT_ROOT))
    model = build_model_from_config(model_yaml)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config_path": rel,
            "train_config_path": "configs/train/wavelet/baseline.yaml",
            "epoch": 3,
            "best_epoch": 2,
            "val_metrics": {
                "accuracy": 0.80,
                "precision": 0.75,
                "recall": 0.70,
                "f1": 0.72,
            },
            "history": [
                {
                    "epoch": 1,
                    "val_acc": 0.70,
                    "val_precision": 0.68,
                    "val_recall": 0.65,
                    "val_f1": 0.66,
                },
                {
                    "epoch": 2,
                    "val_acc": 0.82,
                    "val_precision": 0.79,
                    "val_recall": 0.76,
                    "val_f1": 0.77,
                    "val_auc_roi": 0.88,
                },
            ],
            "classes": ["normal", "anomaly"],
        },
        DUMMY_CKPT,
    )
    return DUMMY_CKPT


def _run_manifest_only_checks(model_yaml: Path) -> None:
    mod = _load_export_module()
    pre = mod.preprocessing_from_pipeline_yaml(mod.DEFAULT_PIPELINE_YAML.resolve())
    ckpt = mod._load_checkpoint(DUMMY_CKPT, torch.device("cpu"))
    cfg_path = model_yaml.resolve()
    model_yaml_dict = mod._load_model_yaml(cfg_path)
    model = build_model_from_config(cfg_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    training = mod._resolve_training_for_manifest(
        ckpt=ckpt,
        train_report=None,
        train_cfg=None,
    )

    manifest = mod._build_gui_manifest(
        base=None,
        preprocessing=pre,
        model=model,
        layers_model=None,
        model_yaml=model_yaml_dict,
        cfg_path=cfg_path,
        ckpt=ckpt,
        input_spec_bands=16,
        patch_h=64,
        patch_w=64,
        onnx_name="model.onnx",
        description="Test manifest-only",
        training=training,
        reducer_method="wavelet",
        embedded_reducer_in_onnx=False,
        reducer_input_bands=16,
        reducer_output_bands=16,
    )

    pre_m = manifest["pipeline"]["preprocessing"]
    pre_expected = mod.preprocessing_from_pipeline_yaml(mod.DEFAULT_PIPELINE_YAML.resolve())
    assert pre_m == pre_expected, "preprocessing skal matche preprocessing_from_pipeline_yaml(DEFAULT_PIPELINE_YAML)"

    assert manifest["artifacts"]["model_onnx"] == "model.onnx"
    assert manifest["pipeline"]["spectral_reducer"]["method"] == "wavelet"
    assert isinstance(manifest["pipeline"]["model"]["layers"], dict)
    assert len(manifest["pipeline"]["model"]["layers"]) > 0
    assert manifest["training"]["metrics"]["accuracy"] == 0.82
    assert manifest["training"]["metrics"]["f1"] == 0.77

    print("[test] manifest-only: OK")
    print("      pipeline.preprocessing.steps:", pre_m["steps"])
    print("      (Matcher configs/preprocessing/pipeline.yaml via preprocessing_from_pipeline_yaml.)")


def _run_full_export(model_yaml: Path) -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--checkpoint",
        str(DUMMY_CKPT),
        "--model-config",
        str(model_yaml),
        "--spectral-bands",
        "16",
        "--patch-h",
        "64",
        "--patch-w",
        "64",
        "--out-dir",
        str(OUT_DIR),
        "--dataset",
        "wavelet_manifest.csv",
        "--reducer-method",
        "wavelet",
        "--train-samples",
        "20544",
        "--description",
        "Test full export (dummy checkpoint)",
    ]
    print("[test] kjører:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    manifest_path = OUT_DIR / "manifest.json"
    assert manifest_path.is_file()
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    mod = _load_export_module()
    assert m["pipeline"]["preprocessing"] == mod.preprocessing_from_pipeline_yaml(
        mod.DEFAULT_PIPELINE_YAML.resolve()
    )
    assert (OUT_DIR / "model.onnx").is_file()

    print("[test] full export: OK")
    print(f"      manifest: {manifest_path}")
    print(f"      onnx:     {OUT_DIR / 'model.onnx'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test export_cnn3d_onnx manifest + valgfri ONNX-eksport.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Kun verifiser manifest-bygging (ingen torch.onnx.export).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Kjør også full ONNX-eksport (krever onnx/onnxscript i venv).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_YAML,
        help="Modell-YAML for dummy-checkpoint (default: baseline_3dcnn).",
    )
    args = parser.parse_args()

    model_yaml = args.model_config
    if not model_yaml.is_absolute():
        model_yaml = (PROJECT_ROOT / model_yaml).resolve()

    if not EXPORT_SCRIPT.is_file():
        print(f"FEIL: Mangler skript: {EXPORT_SCRIPT}", file=sys.stderr)
        return 1

    _write_dummy_checkpoint(model_yaml)
    print(f"[test] dummy checkpoint: {DUMMY_CKPT}")

    _run_manifest_only_checks(model_yaml)

    if args.manifest_only:
        print("\n[test] Ferdig (--manifest-only, ingen ONNX).")
        return 0

    if not args.full:
        print(
            "\n[test] Tips: kjør med --full for å teste ONNX-eksport også "
            "(krever: source .venv/bin/activate).",
        )
        return 0

    try:
        _run_full_export(model_yaml)
    except subprocess.CalledProcessError as e:
        print(
            "\n[test] Full eksport feilet. Sjekk at du har aktivert .venv og at "
            "onnx/onnxscript er installert (pip install onnx onnxscript).",
            file=sys.stderr,
        )
        return e.returncode or 1

    print("\n[test] Alt OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
