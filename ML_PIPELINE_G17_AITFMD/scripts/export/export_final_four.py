#!/usr/bin/env python3
"""
Final-eksport av de fire beste enkeltmodellene til ONNX + manifest for SpectralAssist.

Kjøres fra ML_PIPELINE_G17_AITFMD-roten (med venv aktivert):
    python scripts/export/export_final_four.py [--dry-run]

Utdata:
    outputs/onnx/final/{model_name}/{model_name}_final.onnx
    outputs/onnx/final/{model_name}/manifest.json
    outputs/onnx/final/{model_name}/roi_validation/   (raw/dark/white ENVI-patch)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export" / "export_cnn3d_onnx.py"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Valideringsdata — én ROI brukes for alle fire modeller for konsistens
# ---------------------------------------------------------------------------
VALIDATION_ROI_DIR = (
    PROJECT_ROOT.parent / "PKG - HistologyHSI-GB" / "P10" / "ROI_01_C01_T"
)

# ---------------------------------------------------------------------------
# Fire beste modeller — ett sjekkpunkt per arkitektur, valgt på test-ROI-AUC
# ---------------------------------------------------------------------------
FINAL_MODELS: list[dict] = [
    {
        "out_name": "baseline_3dcnn",
        "onnx_filename": "baseline_3dcnn_final.onnx",
        "checkpoint": "outputs/training/kfold_baseline_3dcnn_20260411_095415/fold_0/best.pt",
        "description": "Baseline 3D-CNN, Wavelet16, best fold_0 (ROI-AUC 0.898 val)",
    },
    {
        "out_name": "resnet_3dcnn",
        "onnx_filename": "resnet_3dcnn_final.onnx",
        "checkpoint": "outputs/training/kfold_resnet_3dcnn_20260419_210955/fold_2/best_by_auc.pt",
        "description": "ResNet 3D-CNN 516K, Wavelet16, best fold_2 (ROI-AUC 0.994)",
    },
    {
        "out_name": "msd_dense",
        "onnx_filename": "msd_dense_final.onnx",
        "checkpoint": "outputs/training/kfold_msd_dense_3dcnn_20260415_195529/fold_2/best.pt",
        "description": "MSD Dense 3D-CNN 83K, Wavelet16, best fold_2 (ROI-AUC 0.963)",
    },
    {
        "out_name": "ad_hybrid_sn",
        "onnx_filename": "ad_hybrid_sn_final.onnx",
        "checkpoint": "outputs/training/kfold_ad_hybrid_sn_3dcnn_20260415_092712/fold_4/best.pt",
        "description": "AD Hybrid SN 3D-CNN 370K, Wavelet16, best fold_4 (ROI-AUC 0.935)",
    },
]

COMMON_ARGS = [
    "--reducer-method", "wavelet",
    "--spectral-bands", "16",
    "--patch-h", "64",
    "--patch-w", "64",
    "--raw-bands", "275",
]


def export_model(model: dict, out_base: Path, dry_run: bool) -> bool:
    ckpt_path = PROJECT_ROOT / model["checkpoint"]
    if not ckpt_path.is_file():
        print(f"  [SKIP] Checkpoint mangler: {ckpt_path.relative_to(PROJECT_ROOT)}")
        return False

    if not VALIDATION_ROI_DIR.is_dir():
        print(f"  [FEIL] Valideringsmappe mangler: {VALIDATION_ROI_DIR}")
        return False

    out_dir = out_base / model["out_name"]

    cmd = [
        PYTHON, str(EXPORT_SCRIPT),
        "--checkpoint", str(ckpt_path),
        "--out-dir", str(out_dir),
        "--onnx-name", model["onnx_filename"],
        "--description", model["description"],
        "--validation-roi-dir", str(VALIDATION_ROI_DIR),
        *COMMON_ARGS,
    ]

    if dry_run:
        print(f"  [DRY] {' '.join(str(c) for c in cmd)}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"  [FEIL] {model['out_name']}:")
        print(result.stderr[-1500:] if result.stderr else "(ingen feilmelding)")
        return False

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")

    import json
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text())
        status = m.get("validation", {}).get("status", "?")
        reducer = m.get("pipeline", {}).get("spectral_reducer", {}).get("method", "?")
        tissue_ratio = m.get("pipeline", {}).get("preprocessing", {}).get("params", {}).get("patch_min_tissue_ratio", "?")
        print(f"  [OK] validation={status}  reducer={reducer}  patch_min_tissue_ratio={tissue_ratio}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksporter de fire beste modellene til ONNX med validering")
    parser.add_argument("--dry-run", action="store_true", help="Skriv ut kommandoer uten å kjøre")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "onnx" / "final",
        help="Rotmappe for utdata (standard: outputs/onnx/final/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Eksporter kun én modell (out_name). Utelat for alle fire.",
    )
    args = parser.parse_args()

    models = FINAL_MODELS
    if args.model is not None:
        models = [m for m in FINAL_MODELS if m["out_name"] == args.model]
        if not models:
            names = [m["out_name"] for m in FINAL_MODELS]
            print(f"Ukjent modell '{args.model}'. Tilgjengelige:\n" + "\n".join(f"  {n}" for n in names))
            sys.exit(1)

    print(f"Valideringsmappe: {VALIDATION_ROI_DIR}")
    print(f"Utdatamappe:      {args.out_dir}\n")

    ok, fail = 0, 0
    for model in models:
        print(f"{'=' * 60}")
        print(f"Modell:     {model['out_name']}")
        print(f"Checkpoint: {model['checkpoint']}")
        print(f"Beskrivelse: {model['description']}")
        success = export_model(model, args.out_dir, args.dry_run)
        if success:
            ok += 1
        else:
            fail += 1
        print()

    print(f"{'=' * 60}")
    print(f"Ferdig: {ok} eksportert, {fail} feilet")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
