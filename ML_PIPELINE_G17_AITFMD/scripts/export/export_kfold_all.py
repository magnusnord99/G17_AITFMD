#!/usr/bin/env python3
"""
Batch-eksport av alle k-fold-kjøringer til ONNX + manifest for SpectralAssist.

Alle k-fold-modeller er trent med Wavelet16 (db2), og eksporteres med
--reducer-method wavelet uavhengig av hva pipeline.yaml sier.

Kjøres fra ML_PIPELINE_G17_AITFMD-roten:
    python scripts/export/export_kfold_all.py [--dry-run] [--run NAVN]

Utdata:
    outputs/onnx_kfold/<run_name>/fold_<N>/model.onnx
    outputs/onnx_kfold/<run_name>/fold_<N>/manifest.json
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
# K-fold kjøringstabell
#
# ckpt_file: hvilken checkpoint-fil som brukes per fold
#   - "best_by_auc.pt" : ROI-AUC-optimalisert (nyere kjøringer, anbefalt)
#   - "best.pt"        : val_loss/eldre kjøringer
#
# num_folds: antall fold (alle er 5 her)
# ---------------------------------------------------------------------------
KFOLD_RUNS: list[dict] = [
    {
        "name": "baseline_3dcnn_wavelet16",
        "description": "Baseline 3D-CNN, Wavelet16, 5-fold (20260411)",
        "run_dir": "outputs/kfold_baseline_3dcnn_20260411_095415",
        "ckpt_file": "best.pt",
        "num_folds": 5,
    },
    {
        "name": "resnet_3dcnn_small_wavelet16",
        "description": "ResNet 3D-CNN liten [8,16,32] 129K param, Wavelet16, 5-fold (20260412)",
        "run_dir": "outputs/kfold_resnet_3dcnn_20260412_074320",
        "ckpt_file": "best.pt",
        "num_folds": 5,
        # ckpt lagret feil model_config_path (stor config) — override til liten
        "model_config": "configs/models/resnet_3dcnn_small.yaml",
    },
    {
        "name": "resnet_3dcnn_wavelet16_20260413",
        "description": "ResNet 3D-CNN stor [16,32,64] 516K param, Wavelet16, 5-fold (20260413)",
        "run_dir": "outputs/kfold_resnet_3dcnn_20260413_103849",
        "ckpt_file": "best.pt",
        "num_folds": 5,
    },
    {
        "name": "resnet_3dcnn_wavelet16_roiauc_20260414",
        "description": "ResNet 3D-CNN 516K, Wavelet16, ROI-AUC checkpoint, 5-fold (20260414)",
        "run_dir": "outputs/kfold_resnet_3dcnn_20260414_101159",
        "ckpt_file": "best_by_auc.pt",
        "num_folds": 5,
    },
    {
        "name": "resnet_3dcnn_wavelet16_roiauc_20260419",
        "description": "ResNet 3D-CNN 516K, Wavelet16, ROI-AUC checkpoint, 5-fold (20260419)",
        "run_dir": "outputs/kfold_resnet_3dcnn_20260419_210955",
        "ckpt_file": "best_by_auc.pt",
        "num_folds": 5,
    },
    {
        "name": "resnet_3dcnn_wavelet16_roiauc_20260420",
        "description": "ResNet 3D-CNN 516K, Wavelet16, ROI-AUC checkpoint, 5-fold (20260420)",
        "run_dir": "outputs/kfold_resnet_3dcnn_20260420_065029",
        "ckpt_file": "best_by_auc.pt",
        "num_folds": 5,
    },
    {
        "name": "ad_hybrid_sn_wavelet16_20260415",
        "description": "AD Hybrid SN 3D-CNN 370K, Wavelet16, 5-fold (20260415)",
        "run_dir": "outputs/kfold_ad_hybrid_sn_3dcnn_20260415_092712",
        "ckpt_file": "best.pt",
        "num_folds": 5,
    },
    {
        "name": "ad_hybrid_sn_wavelet16_20260422",
        "description": "AD Hybrid SN 3D-CNN 370K, Wavelet16, 5-fold (20260422)",
        "run_dir": "outputs/kfold_ad_hybrid_sn_3dcnn_20260422_122652",
        "ckpt_file": "best.pt",
        "num_folds": 5,
    },
    {
        "name": "msd_dense_3dcnn_wavelet16",
        "description": "MSD Dense 3D-CNN 83K, Wavelet16, 5-fold (20260415)",
        "run_dir": "outputs/kfold_msd_dense_3dcnn_20260415_195529",
        "ckpt_file": "best.pt",
        "num_folds": 5,
    },
]

# Felles eksport-parametere for alle wavelet-trente kfold-modeller
COMMON_EXPORT_ARGS = [
    "--reducer-method", "wavelet",  # eksplisitt — ikke les fra pipeline.yaml
    "--spectral-bands", "16",
    "--patch-h", "64",
    "--patch-w", "64",
    "--raw-bands", "275",
]


def export_fold(
    run: dict,
    fold_idx: int,
    out_base: Path,
    dry_run: bool,
) -> bool:
    """Eksporter én fold. Returnerer True ved suksess."""
    run_dir = PROJECT_ROOT / run["run_dir"]
    fold_dir = run_dir / f"fold_{fold_idx}"
    ckpt_path = fold_dir / run["ckpt_file"]

    if not ckpt_path.is_file():
        print(f"  [SKIP] {ckpt_path.relative_to(PROJECT_ROOT)} finnes ikke")
        return False

    out_dir = out_base / run["name"] / f"fold_{fold_idx}"
    onnx_name = f"{run['name']}_fold{fold_idx}.onnx"

    cmd = [
        PYTHON, str(EXPORT_SCRIPT),
        "--checkpoint", str(ckpt_path),
        "--out-dir", str(out_dir),
        "--onnx-name", onnx_name,
        "--description", f"{run['description']} — fold {fold_idx}",
        *COMMON_EXPORT_ARGS,
    ]

    if "model_config" in run:
        cmd += ["--model-config", str(PROJECT_ROOT / run["model_config"])]

    if dry_run:
        print(f"  [DRY] {' '.join(str(c) for c in cmd)}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"  [FEIL] fold_{fold_idx}:")
        print(result.stderr[-1000:] if result.stderr else "(ingen feilmelding)")
        return False

    # Verifiser at reducer ble satt riktig
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        import json
        manifest = json.loads(manifest_path.read_text())
        reducer = manifest.get("pipeline", {}).get("spectral_reducer", {}).get("method", "?")
        if reducer != "wavelet":
            print(f"  [ADVARSEL] manifest.reducer={reducer} — forventet wavelet!")
        else:
            print(f"  [OK] fold_{fold_idx} → {onnx_name}  reducer={reducer}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-eksport av alle k-fold-modeller til ONNX")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skriv ut kommandoer uten å kjøre dem",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Eksporter kun én kjøring (navn fra KFOLD_RUNS). Utelat for alle.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "onnx_kfold",
        help="Utdatakatalog (standard: outputs/onnx_kfold/)",
    )
    args = parser.parse_args()

    runs = KFOLD_RUNS
    if args.run is not None:
        runs = [r for r in KFOLD_RUNS if r["name"] == args.run]
        if not runs:
            names = [r["name"] for r in KFOLD_RUNS]
            print(f"Ukjent run '{args.run}'. Tilgjengelige:\n" + "\n".join(f"  {n}" for n in names))
            sys.exit(1)

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for run in runs:
        print(f"\n{'='*60}")
        print(f"Kjøring: {run['name']}")
        print(f"  {run['description']}")
        print(f"  Checkpoint: {run['ckpt_file']}")
        print(f"  Reducer:    wavelet (eksplisitt, ignorerer pipeline.yaml)")

        for fold_idx in range(run["num_folds"]):
            ok = export_fold(run, fold_idx, args.out_dir, args.dry_run)
            if ok:
                total_ok += 1
            else:
                # Distinguish skip vs fail based on file existence
                run_dir = PROJECT_ROOT / run["run_dir"]
                ckpt = run_dir / f"fold_{fold_idx}" / run["ckpt_file"]
                if not ckpt.is_file():
                    total_skip += 1
                else:
                    total_fail += 1

    print(f"\n{'='*60}")
    print(f"Ferdig: {total_ok} eksportert, {total_skip} hoppet over, {total_fail} feilet")
    if total_ok > 0 and not args.dry_run:
        print(f"Utdata: {args.out_dir}")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
