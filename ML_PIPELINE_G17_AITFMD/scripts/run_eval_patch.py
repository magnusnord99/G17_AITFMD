#!/usr/bin/env python3
"""
Patch-nivå evaluering: samme rutenett og maske-filter som run_eval.py,
men hver patch får ROI-ens label_id (som under trening). Metrikker over alle patcher.

Patcher fra samme ROI er ikke statistisk uavhengige — bruk som supplement til ROI-eval.

Bruk:
  python scripts/run_eval_patch.py --checkpoint ..._best.pt --config configs/train.yaml --split test
  python scripts/run_eval_patch.py --checkpoint ..._best.pt \\
    --hyperparams outputs/plots/resnet_3dcnn/wavelet16/RUN_ID/hyperparams.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.inference import InferenceEngine
from src.evaluation.metrics import compute_eval_metrics
from src.models import build_model_from_config
from src.preprocessing.patching import iter_patches, load_mask, load_numpy_cube
from src.utils.logging_setup import configure_logging, get_logger


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_cfg_from_hyperparams(hp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw: dict[str, Any] = yaml.safe_load(hp_path.read_text(encoding="utf-8"))
    tcp = raw.get("train_config_parsed")
    if not isinstance(tcp, dict):
        raise ValueError(f"train_config_parsed missing or invalid in {hp_path}")
    return tcp, raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch-level eval (all grid patches inherit ROI label_id).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *_best.pt from run_train")
    parser.add_argument("--config", type=str, default=None, help="Train YAML")
    parser.add_argument("--hyperparams", type=str, default=None, help="Path to hyperparams.yaml from run_train")
    parser.add_argument("--split", type=str, choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None, help="Report path")
    args = parser.parse_args()

    if args.config is None and not args.hyperparams:
        args.config = "configs/train.yaml"

    cfg: dict[str, Any]
    hp_raw: dict[str, Any] | None = None
    hp_path: Path | None = None

    if args.hyperparams:
        hp_path = Path(args.hyperparams).expanduser().resolve()
        if not hp_path.exists():
            hp_path = (PROJECT_ROOT / args.hyperparams).resolve()
        if not hp_path.exists():
            print(f"[patch_eval] ERROR: hyperparams not found: {args.hyperparams}")
            return 1
        cfg, hp_raw = _load_cfg_from_hyperparams(hp_path)
        resolved = hp_raw.get("resolved_paths") or {}
        data_cfg = dict(cfg.get("data", {}))
        if resolved.get("manifest"):
            data_cfg["cube_manifest_csv"] = resolved["manifest"]
        if resolved.get("cube_root"):
            data_cfg["cube_root"] = resolved["cube_root"]
        if resolved.get("mask_root"):
            data_cfg["mask_root"] = resolved["mask_root"]
        cfg = {**cfg, "data": data_cfg}
        run_id = hp_raw.get("run_id", "patch_eval")
        out_dir = hp_path.parent
    else:
        assert args.config is not None
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            config_path = (PROJECT_ROOT / args.config).resolve()
        if not config_path.exists():
            print(f"[patch_eval] ERROR: config not found: {args.config}")
            return 1
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_id = "patch_eval"
        paths_cfg = cfg.get("paths", {})
        out_dir = (PROJECT_ROOT / str(paths_cfg.get("plots_dir", "outputs/plots"))).resolve()

    configure_logging(log_dir=out_dir / "logs", run_name=f"patch_eval_{args.split}_{run_id}")
    log = get_logger(__name__)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        ckpt_path = (PROJECT_ROOT / args.checkpoint).resolve()
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    data_cfg = cfg.get("data", {})
    for k in ("patch_h", "patch_w", "stride_h", "stride_w"):
        if k not in data_cfg:
            log.error("data.%s missing in config", k)
            return 1

    patch_h = int(data_cfg["patch_h"])
    patch_w = int(data_cfg["patch_w"])
    stride_h = int(data_cfg["stride_h"])
    stride_w = int(data_cfg["stride_w"])
    min_tissue = float(data_cfg.get("min_tissue_ratio", 0.0))
    batch_size = int(args.batch_size or data_cfg.get("batch_size", 16))

    def _res(raw: str) -> Path:
        p = Path(raw).expanduser()
        return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    manifest_path = _res(str(data_cfg.get("cube_manifest_csv", "")))
    cube_root_raw = data_cfg.get("cube_root")
    cube_root = _res(str(cube_root_raw)) if cube_root_raw else None
    if cube_root and not cube_root.exists():
        cube_root = None
    mask_root_raw = data_cfg.get("mask_root")
    mask_root = _res(str(mask_root_raw)) if mask_root_raw else None
    if mask_root and not mask_root.exists():
        mask_root = None

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return 1

    df = pd.read_csv(manifest_path)
    split_df = df[df["split"] == args.split].reset_index(drop=True)
    if len(split_df) == 0:
        log.error("No rows with split=%r", args.split)
        return 1

    device = _pick_device()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg_path = Path(str(ckpt.get("model_config_path", "")))
    if not model_cfg_path.is_absolute():
        model_cfg_path = (PROJECT_ROOT / model_cfg_path).resolve()
    if not model_cfg_path.exists():
        log.error("model_config_path not found: %s", model_cfg_path)
        return 1

    model = build_model_from_config(model_cfg_path).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    engine = InferenceEngine(model, device, batch_size=batch_size, amp=False)

    y_true_all: list[int] = []
    y_prob_all: list[float] = []
    y_pred_all: list[int] = []
    per_roi: list[dict[str, Any]] = []
    skipped = 0

    for idx in range(len(split_df)):
        row = split_df.iloc[idx]
        roi_id = f"{row['patient_id']}/{row['roi_name']}"

        if cube_root and "patient_id" in row and "roi_name" in row:
            cube_path = cube_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        else:
            cube_path = Path(str(row["output_path"]))

        if not cube_path.exists():
            log.warning("Skipping missing cube: %s", cube_path)
            skipped += 1
            continue

        cube = load_numpy_cube(cube_path)
        mask = None
        if mask_root and "patient_id" in row and "roi_name" in row:
            mp = mask_root / str(row["patient_id"]) / f"{row['roi_name']}_mask.npy"
            if mp.exists():
                try:
                    mask = load_mask(mp)
                except Exception:
                    pass

        patches: list[np.ndarray] = []
        for patch, _y, _x in iter_patches(
            cube, patch_h=patch_h, patch_w=patch_w,
            stride_h=stride_h, stride_w=stride_w,
            mask=mask, min_tissue_ratio=min_tissue,
        ):
            patches.append(patch)

        if not patches:
            log.warning("No valid patches for ROI %s", roi_id)
            skipped += 1
            continue

        lab = int(row["label_id"])
        labels = [lab] * len(patches)
        result = engine.run_on_patches(patches, labels=labels, roi_id=roi_id)

        y_true_all.extend(labels)
        y_prob_all.extend(result.probs.tolist())
        y_pred_all.extend(result.preds.tolist())

        n = len(patches)
        correct = int(np.sum(result.preds == lab))
        per_roi.append({
            "patient_id": str(row["patient_id"]),
            "roi_name": str(row["roi_name"]),
            "label_id": lab,
            "n_patches": n,
            "n_correct": correct,
            "patch_accuracy": float(correct / max(1, n)),
        })

    if not y_true_all:
        log.error("No patches evaluated")
        return 1

    patch_metrics = compute_eval_metrics(
        np.asarray(y_true_all), np.asarray(y_prob_all), threshold=0.5
    )

    report: dict[str, Any] = {
        "eval_level": "patch",
        "note": "Each patch uses its ROI label_id. Patches within an ROI are correlated.",
        "checkpoint": str(ckpt_path.resolve()),
        "split": args.split,
        "patch_h": patch_h, "patch_w": patch_w,
        "stride_h": stride_h, "stride_w": stride_w,
        "min_tissue_ratio": min_tissue,
        "n_patches_total": len(y_true_all),
        "n_rois_with_patches": len(per_roi),
        "n_rois_skipped": skipped,
        "device": device.type,
        "patch_metrics": patch_metrics,
        "per_roi": per_roi,
    }
    if hp_raw is not None:
        report["run_id"] = hp_raw.get("run_id")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.out_json:
        outp = Path(args.out_json).expanduser().resolve()
    else:
        outp_dir = hp_path.parent if hp_path else out_dir
        outp_dir.mkdir(parents=True, exist_ok=True)
        outp = outp_dir / f"patch_eval_{args.split}_{ts}.json"

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    m = patch_metrics
    log.info("split=%s  patches=%d  ROIs=%d  skipped=%d",
             args.split, len(y_true_all), len(per_roi), skipped)
    log.info("acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc_roc=%.4f",
             m.get("accuracy", 0), m.get("precision", 0),
             m.get("recall", 0), m.get("f1", 0), m.get("auc_roc", float("nan")))
    log.info("Report: %s", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
