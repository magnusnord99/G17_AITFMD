#!/usr/bin/env python3
"""
ROI-nivå (og valgfritt pasient-nivå) evaluering av en trent checkpoint.

Bruker samme patch-geometri som train.yaml (patch_h/w, stride_h/w, mask, min_tissue_ratio).
For hver ROI (kube-rad i manifest): kjør modellen på alle gyldige patch-posisjoner i rutenettet,
aggreger til én prediksjon per ROI (gjennomsnitt av P(klasse=1) eller majoritetsstem blant patch-klasser),
sammenlign med label_id.

Bruk:
  python scripts/run_eval.py --checkpoint outputs/checkpoints/baseline_3dcnn_RUN_best.pt --config configs/train.yaml
  python scripts/run_eval.py --checkpoint path/to/best.pt --config path/to/train.yaml --split test --patient-level

Krever at checkpoint er laget av run_train.py (model_config_path i checkpoint).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model_from_config
from src.preprocessing.patching import iter_patches, load_mask, load_numpy_cube


def _resolve_path(config_path: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    candidate = (config_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / p).resolve()


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """(H,W,C) -> (1,1,C,H,W) for Conv3d batching (samme logikk som CubePatchDataset + stack)."""
    return torch.from_numpy(patch.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)


def _load_mask_for_row(mask_root: Path | None, row: pd.Series) -> np.ndarray | None:
    if not mask_root or "patient_id" not in row or "roi_name" not in row:
        return None
    mask_path = mask_root / str(row["patient_id"]) / f"{row['roi_name']}_mask.npy"
    if not mask_path.exists():
        return None
    try:
        return load_mask(mask_path)
    except Exception:
        return None


def _resolve_cube_path(row: pd.Series, cube_root: Path | None) -> Path:
    if cube_root and "patient_id" in row and "roi_name" in row:
        return cube_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
    return Path(str(row["output_path"]))


@torch.no_grad()
def _forward_batches(
    model: nn.Module,
    device: torch.device,
    patches: list[np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returner (probs_class1, pred_class) per patch, shape (N,) each."""
    probs = []
    preds = []
    for i in range(0, len(patches), batch_size):
        chunk = patches[i : i + batch_size]
        # Samme som DataLoader: stack av (1,C,H,W) -> (B,1,C,H,W); cat på dim 0 gir feil (B,C,H,W).
        batch = torch.stack([_patch_to_tensor(p) for p in chunk], dim=0).to(device)
        logits = model(batch)
        pr = torch.softmax(logits, dim=1)
        p1 = pr[:, 1].cpu().numpy()
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        probs.append(p1)
        preds.append(pred)
    return np.concatenate(probs), np.concatenate(preds)


def _binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / total
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _aggregate_roi_metrics(
    y_true: list[int],
    y_pred_mean: list[int],
    y_pred_majority: list[int],
) -> dict[str, Any]:
    def counts(yt: list[int], yp: list[int]) -> dict[str, int]:
        tp = tn = fp = fn = 0
        for t, p in zip(yt, yp, strict=True):
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 0:
                tn += 1
            elif t == 0 and p == 1:
                fp += 1
            else:
                fn += 1
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    c_m = counts(y_true, y_pred_mean)
    c_v = counts(y_true, y_pred_majority)
    return {
        "mean_prob_threshold_0.5": {**c_m, **_binary_metrics_from_counts(c_m["tp"], c_m["tn"], c_m["fp"], c_m["fn"])},
        "majority_vote_patches": {**c_v, **_binary_metrics_from_counts(c_v["tp"], c_v["tn"], c_v["fp"], c_v["fn"])},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI-level (and optional patient-level) eval from checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *_best.pt or .pt from run_train")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Same train.yaml as training (data + paths)")
    parser.add_argument("--split", type=str, choices=("val", "test"), default="val", help="Which manifest split to evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size for inference")
    parser.add_argument(
        "--patient-level",
        action="store_true",
        help="Also aggregate ROI predictions to patient_id (majority over ROIs; label must agree per patient)",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Write report JSON to this path")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[eval] ERROR: config not found: {args.config}", flush=True)
        return 1

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_absolute() or not ckpt_path.exists():
        ckpt_path = (PROJECT_ROOT / args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"[eval] ERROR: checkpoint not found: {args.checkpoint}", flush=True)
        return 1

    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    for k in ("patch_h", "patch_w", "stride_h", "stride_w"):
        if k not in data_cfg:
            print(f"[eval] ERROR: data.{k} missing in {config_path}", flush=True)
            return 1

    patch_h = int(data_cfg["patch_h"])
    patch_w = int(data_cfg["patch_w"])
    stride_h = int(data_cfg["stride_h"])
    stride_w = int(data_cfg["stride_w"])
    min_tissue = float(data_cfg.get("min_tissue_ratio", 0.0))
    batch_size = int(args.batch_size if args.batch_size is not None else data_cfg.get("batch_size", 16))

    manifest_path = _resolve_path(config_path, str(data_cfg.get("cube_manifest_csv", "")))
    if not manifest_path.exists():
        print(f"[eval] ERROR: manifest not found: {manifest_path}", flush=True)
        return 1

    mask_root_raw = data_cfg.get("mask_root")
    mask_path = _resolve_path(config_path, str(mask_root_raw)) if mask_root_raw else None
    mask_root = mask_path if (mask_path and mask_path.exists()) else None

    cube_root_raw = data_cfg.get("cube_root")
    cube_root = _resolve_path(config_path, str(cube_root_raw)) if cube_root_raw else None
    if cube_root and not cube_root.exists():
        cube_root = None

    df = pd.read_csv(manifest_path)
    split_df = df[df["split"] == args.split].reset_index(drop=True)
    if len(split_df) == 0:
        print(f"[eval] ERROR: no rows with split={args.split!r}", flush=True)
        return 1

    if "patient_id" not in split_df.columns or "roi_name" not in split_df.columns:
        print("[eval] ERROR: manifest needs patient_id and roi_name for ROI eval", flush=True)
        return 1

    device = _pick_device()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg_path = ckpt.get("model_config_path")
    if not model_cfg_path:
        print("[eval] ERROR: checkpoint missing model_config_path (use checkpoints from run_train.py)", flush=True)
        return 1
    model_cfg_path = Path(model_cfg_path)
    if not model_cfg_path.is_absolute():
        model_cfg_path = (PROJECT_ROOT / model_cfg_path).resolve()

    model = build_model_from_config(model_cfg_path).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    roi_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred_mean: list[int] = []
    y_pred_majority: list[int] = []

    skipped = 0
    for idx in range(len(split_df)):
        row = split_df.iloc[idx]
        cube_path = _resolve_cube_path(row, cube_root)
        if not cube_path.exists():
            print(f"[eval] WARN: skip missing cube: {cube_path}", flush=True)
            skipped += 1
            continue

        cube = load_numpy_cube(cube_path)
        mask = _load_mask_for_row(mask_root, row)
        patches: list[np.ndarray] = []
        for patch, _y, _x in iter_patches(
            cube,
            patch_h=patch_h,
            patch_w=patch_w,
            stride_h=stride_h,
            stride_w=stride_w,
            mask=mask,
            min_tissue_ratio=min_tissue,
        ):
            patches.append(patch)

        if not patches:
            print(f"[eval] WARN: no valid patches for ROI {row.get('patient_id')}/{row.get('roi_name')}", flush=True)
            skipped += 1
            continue

        p1, pred_cls = _forward_batches(model, device, patches, batch_size=batch_size)
        mean_p1 = float(np.mean(p1))
        pred_mean = 1 if mean_p1 >= 0.5 else 0
        maj = int(np.argmax(np.bincount(pred_cls.astype(int), minlength=2)))
        true_label = int(row["label_id"])

        roi_rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "roi_name": str(row["roi_name"]),
                "n_patches": len(patches),
                "mean_prob_class_1": mean_p1,
                "pred_mean_prob": pred_mean,
                "pred_majority": maj,
                "label_id": true_label,
            }
        )
        y_true.append(true_label)
        y_pred_mean.append(pred_mean)
        y_pred_majority.append(maj)

    if not y_true:
        print("[eval] ERROR: no ROIs evaluated", flush=True)
        return 1

    metrics = _aggregate_roi_metrics(y_true, y_pred_mean, y_pred_majority)
    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path.resolve()),
        "train_config": str(config_path.resolve()),
        "split": args.split,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "n_rois_evaluated": len(y_true),
        "n_rois_skipped": skipped,
        "device": device.type,
        "roi_metrics": metrics,
        "per_roi": roi_rows,
    }

    print("\n[eval] ========== ROI-nivå (én prediksjon per ROI) ==========", flush=True)
    print(f"[eval] Checkpoint: {ckpt_path}", flush=True)
    print(f"[eval] Config:      {config_path}", flush=True)
    print(f"[eval] Split:       {args.split}  (n={len(y_true)} ROI, skipped={skipped})", flush=True)
    print(f"[eval] Patch grid:  {patch_h}x{patch_w} stride {stride_h}x{stride_w}", flush=True)

    for name, m in metrics.items():
        print(
            f"\n[eval] {name}:",
            f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  "
            f"(tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']})",
            flush=True,
        )

    if args.patient_level:
        by_p: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in roi_rows:
            by_p[r["patient_id"]].append(r)

        pt_true: list[int] = []
        pt_pred_m: list[int] = []
        pt_pred_v: list[int] = []

        for pid, rows in sorted(by_p.items()):
            labels = {int(r["label_id"]) for r in rows}
            if len(labels) != 1:
                print(f"[eval] WARN: patient {pid} has inconsistent label_id across ROIs: {labels}", flush=True)
            true_p = int(rows[0]["label_id"])
            # majoritet over ROI-prediksjoner (mean_prob-basert og majoritet-basert)
            pm = [r["pred_mean_prob"] for r in rows]
            pv = [r["pred_majority"] for r in rows]
            pred_m = 1 if sum(pm) >= len(pm) / 2 else 0  # majority of ROI votes
            pred_v = 1 if sum(pv) >= len(pv) / 2 else 0
            pt_true.append(true_p)
            pt_pred_m.append(pred_m)
            pt_pred_v.append(pred_v)

        c_pm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for t, p in zip(pt_true, pt_pred_m, strict=True):
            if t == 1 and p == 1:
                c_pm["tp"] += 1
            elif t == 0 and p == 0:
                c_pm["tn"] += 1
            elif t == 0 and p == 1:
                c_pm["fp"] += 1
            else:
                c_pm["fn"] += 1
        c_pv = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for t, p in zip(pt_true, pt_pred_v, strict=True):
            if t == 1 and p == 1:
                c_pv["tp"] += 1
            elif t == 0 and p == 0:
                c_pv["tn"] += 1
            elif t == 0 and p == 1:
                c_pv["fp"] += 1
            else:
                c_pv["fn"] += 1

        met_pm = {**c_pm, **_binary_metrics_from_counts(c_pm["tp"], c_pm["tn"], c_pm["fp"], c_pm["fn"])}
        met_pv = {**c_pv, **_binary_metrics_from_counts(c_pv["tp"], c_pv["tn"], c_pv["fp"], c_pv["fn"])}
        report["patient_metrics"] = {
            "majority_of_roi_mean_prob": met_pm,
            "majority_of_roi_majority_patch": met_pv,
            "n_patients": len(by_p),
        }

        print("\n[eval] ========== Pasient-nivå (majoritetsstem over ROI-er) ==========", flush=True)
        print(
            f"[eval] patients={len(by_p)}  mean_prob vote: acc={met_pm['accuracy']:.4f} f1={met_pm['f1']:.4f}",
            flush=True,
        )
        print(
            f"[eval] patients={len(by_p)}  majority vote:  acc={met_pv['accuracy']:.4f} f1={met_pv['f1']:.4f}",
            flush=True,
        )

    out_path = args.out_json
    if out_path:
        outp = Path(out_path).expanduser().resolve()
    else:
        reports_dir = _resolve_path(config_path, str(cfg.get("paths", {}).get("reports_dir", "outputs/reports")))
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        outp = reports_dir / f"roi_eval_{args.split}_{ts}.json"

    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[eval] Report written: {outp}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
