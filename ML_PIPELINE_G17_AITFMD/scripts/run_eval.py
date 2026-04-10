#!/usr/bin/env python3
"""
ROI-nivå (og valgfritt pasient-nivå) evaluering av en trent checkpoint.

Bruker samme patch-geometri som train.yaml (patch_h/w, stride_h/w, mask, min_tissue_ratio).
Nå med AUC-ROC, PR-kurver, og Youden-optimal terskelvalg.

Bruk:
  python scripts/run_eval.py --checkpoint outputs/checkpoints/baseline_3dcnn_RUN_best.pt \\
      --config configs/train.yaml
  python scripts/run_eval.py --checkpoint ..._best.pt \\
      --hyperparams outputs/plots/resnet_3dcnn/wavelet16/RUN_ID/hyperparams.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator
from src.utils.logging_setup import configure_logging, get_logger


def _load_cfg_from_hyperparams(hp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw: dict[str, Any] = yaml.safe_load(hp_path.read_text(encoding="utf-8"))
    tcp = raw.get("train_config_parsed")
    if not isinstance(tcp, dict):
        raise ValueError(f"train_config_parsed missing or invalid in {hp_path}")
    return tcp, raw


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI-level eval from checkpoint (AUC-ROC, PR curves).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to *_best.pt from run_train")
    parser.add_argument("--config", type=str, default=None, help="Train YAML (data + paths)")
    parser.add_argument(
        "--hyperparams", type=str, default=None,
        help="Path to outputs/plots/.../RUN_ID/hyperparams.yaml (preferred — uses resolved paths from training)",
    )
    parser.add_argument("--split", type=str, choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size for inference")
    parser.add_argument("--patient-level", action="store_true", help="Also compute patient-level metrics")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for reports and plots")
    args = parser.parse_args()

    if args.config is None and not args.hyperparams:
        args.config = "configs/train.yaml"

    cfg: dict[str, Any]
    hp_raw: dict[str, Any] | None = None

    if args.hyperparams:
        hp_path = Path(args.hyperparams).expanduser().resolve()
        if not hp_path.exists():
            hp_path = (PROJECT_ROOT / args.hyperparams).resolve()
        if not hp_path.exists():
            print(f"[eval] ERROR: hyperparams not found: {args.hyperparams}")
            return 1
        cfg, hp_raw = _load_cfg_from_hyperparams(hp_path)
        # Restore resolved paths into cfg.data so Evaluator can find cubes
        resolved = hp_raw.get("resolved_paths") or {}
        data_cfg = dict(cfg.get("data", {}))
        if resolved.get("manifest"):
            data_cfg["cube_manifest_csv"] = resolved["manifest"]
        if resolved.get("cube_root"):
            data_cfg["cube_root"] = resolved["cube_root"]
        if resolved.get("mask_root"):
            data_cfg["mask_root"] = resolved["mask_root"]
        cfg = {**cfg, "data": data_cfg}
        run_id = hp_raw.get("run_id", "eval")
        out_dir = Path(args.out_dir) if args.out_dir else hp_path.parent
    else:
        assert args.config is not None
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            config_path = (PROJECT_ROOT / args.config).resolve()
        if not config_path.exists():
            print(f"[eval] ERROR: config not found: {args.config}")
            return 1
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_id = "eval"
        paths_cfg = cfg.get("paths", {})
        out_dir = Path(args.out_dir) if args.out_dir else (
            PROJECT_ROOT / str(paths_cfg.get("plots_dir", "outputs/plots"))
        ).resolve()

    configure_logging(log_dir=out_dir / "logs", run_name=f"eval_{args.split}_{run_id}")
    log = get_logger(__name__)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        ckpt_path = (PROJECT_ROOT / args.checkpoint).resolve()
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    evaluator = Evaluator(
        checkpoint_path=ckpt_path,
        cfg={**cfg, "_run_id": run_id},
        output_dir=out_dir,
        batch_size=args.batch_size,
        patient_level=args.patient_level,
    )

    metrics = evaluator.run(args.split)

    auc = metrics.get("auc_roc", float("nan"))
    ap = metrics.get("avg_precision", float("nan"))
    opt_thr = metrics.get("optimal_threshold_youden", 0.5)
    m05 = metrics.get("metrics_at_threshold_0.5", {})
    log.info("split=%s  auc_roc=%.4f  avg_precision=%.4f  opt_threshold=%.4f",
             args.split, auc, ap, opt_thr)
    log.info("@0.5: acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
             m05.get("accuracy", 0), m05.get("precision", 0),
             m05.get("recall", 0), m05.get("f1", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
