"""Evaluator: orchestrates one full eval run for a single split."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.aggregation import (
    majority_vote_aggregate,
    mean_prob_aggregate,
    patient_level_aggregate,
)
from src.evaluation.inference import InferenceEngine, InferenceResult
from src.evaluation.metrics import (
    compute_eval_metrics,
    compute_pr,
    compute_roc,
    find_optimal_threshold,
)
from src.evaluation.reporting import (
    append_eval_summary_csv,
    plot_pr_curve,
    plot_roc_curve,
    save_eval_json,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Evaluator:
    """Last checkpoint og kjør ROI-nivå evaluering for ett datasplit.

    Bruk: ev = Evaluator(checkpoint_path, cfg, output_dir); metrics = ev.run("val")
    """

    def __init__(
        self,
        checkpoint_path: Path,
        cfg: dict[str, Any],
        output_dir: Path,
        batch_size: int | None = None,
        patient_level: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.patient_level = patient_level

        data_cfg = cfg.get("data", {})
        self.batch_size = batch_size or int(data_cfg.get("batch_size", 16))

        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def _load_model(self) -> tuple[torch.nn.Module, torch.device]:
        if self._model is not None and self._device is not None:
            return self._model, self._device

        from src.models import build_model_from_config  # type: ignore

        from src.utils.device import pick_training_device
        device = pick_training_device()

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.checkpoint_path, map_location=device)

        model_cfg_path = ckpt.get("model_config_path")
        if not model_cfg_path:
            raise ValueError(
                f"Checkpoint {self.checkpoint_path} is missing 'model_config_path'. "
                "Only checkpoints produced by run_train.py are supported."
            )
        model_cfg_path = Path(model_cfg_path)
        if not model_cfg_path.is_absolute():
            model_cfg_path = (PROJECT_ROOT / model_cfg_path).resolve()

        model = build_model_from_config(model_cfg_path).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        self._model = model
        self._device = device
        log.info("Loaded checkpoint: %s → device=%s", self.checkpoint_path.name, device.type)
        return model, device

    def run(self, split: str = "val") -> dict[str, Any]:
        """Evaluer checkpoint på gitt split ("val" eller "test"). Returner metrikk-dict."""
        log.info("--- Evaluating split=%s ---", split)

        model, device = self._load_model()
        data_cfg = self.cfg.get("data", {})

        manifest_path, mask_root, cube_root = self._resolve_data_paths(data_cfg)
        df = pd.read_csv(manifest_path)
        split_df = df[df["split"] == split].reset_index(drop=True)

        if len(split_df) == 0:
            raise ValueError(f"No rows with split={split!r} in {manifest_path}")

        patch_cfg = {
            "patch_h": data_cfg["patch_h"],
            "patch_w": data_cfg["patch_w"],
            "stride_h": data_cfg["stride_h"],
            "stride_w": data_cfg["stride_w"],
            "min_tissue_ratio": data_cfg.get("min_tissue_ratio", 0.0),
        }

        engine = InferenceEngine(model, device, batch_size=self.batch_size, amp=False)

        per_roi: list[dict[str, Any]] = []
        y_true: list[int] = []
        y_prob_all: list[float] = []
        y_pred_mean_all: list[int] = []
        y_pred_majority_all: list[int] = []
        skipped = 0

        for idx in range(len(split_df)):
            row = split_df.iloc[idx]
            roi_id = f"{row['patient_id']}/{row['roi_name']}"
            cube_path = self._resolve_cube_path(row, cube_root)

            if not cube_path.exists():
                log.warning("Skipping missing cube: %s", cube_path)
                skipped += 1
                continue

            mask_path = self._resolve_mask_path(row, mask_root)
            true_label = int(row["label_id"])

            result: InferenceResult = engine.run_on_roi(
                cube_path,
                patch_cfg,
                label=true_label,
                mask_path=mask_path,
                roi_id=roi_id,
            )

            if result.n_patches == 0:
                log.warning("No valid patches for ROI %s", roi_id)
                skipped += 1
                continue

            agg_mean = mean_prob_aggregate(result)
            agg_maj = majority_vote_aggregate(result)

            roi_record = {
                "patient_id": str(row["patient_id"]),
                "roi_name": str(row["roi_name"]),
                "true_label": true_label,
                "n_patches": result.n_patches,
                "mean_prob_class_1": agg_mean["mean_prob"],
                "pred_mean_prob": agg_mean["pred_label"],
                "pred_majority": agg_maj["pred_label"],
            }
            per_roi.append(roi_record)
            y_true.append(true_label)
            y_prob_all.append(agg_mean["mean_prob"])
            y_pred_mean_all.append(agg_mean["pred_label"])
            y_pred_majority_all.append(agg_maj["pred_label"])

        if not y_true:
            raise RuntimeError(f"No ROIs evaluated for split={split}")

        log.info("Evaluated %d ROIs (%d skipped)", len(y_true), skipped)

        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_prob_arr = np.asarray(y_prob_all, dtype=np.float32)

        # Metrics at 0.5
        metrics_05 = compute_eval_metrics(y_true_arr, y_prob_arr, threshold=0.5)

        # Optimal threshold via Youden's J
        opt_threshold = 0.5
        if len(np.unique(y_true_arr)) == 2:
            fpr, tpr, thresholds = compute_roc(y_true_arr, y_prob_arr)
            opt_threshold = find_optimal_threshold(fpr, tpr, thresholds, strategy="youden")

        metrics_opt = compute_eval_metrics(y_true_arr, y_prob_arr, threshold=opt_threshold)

        # Also compute majority-vote metrics at 0.5
        metrics_majority = compute_eval_metrics(
            y_true_arr,
            np.asarray(y_pred_majority_all, dtype=np.float32),
            threshold=0.5,
        )

        summary_metrics: dict[str, Any] = {
            "auc_roc": metrics_05.get("auc_roc", float("nan")),
            "avg_precision": metrics_05.get("avg_precision", float("nan")),
            "optimal_threshold_youden": opt_threshold,
            "metrics_at_threshold_0.5": metrics_05,
            "metrics_at_optimal_threshold": metrics_opt,
            "majority_vote_metrics": metrics_majority,
            "n_rois_evaluated": len(y_true),
            "n_rois_skipped": skipped,
        }

        # Patient-level
        if self.patient_level:
            pt_results_mean = patient_level_aggregate(per_roi, strategy="mean_prob")
            pt_results_maj = patient_level_aggregate(per_roi, strategy="majority_vote")
            pt_true = [r["true_label"] for r in pt_results_mean]
            pt_pred_m = [r["pred_label"] for r in pt_results_mean]
            pt_pred_v = [r["pred_label"] for r in pt_results_maj]
            summary_metrics["patient_metrics"] = {
                "n_patients": len(pt_results_mean),
                "mean_prob_vote": compute_eval_metrics(
                    np.asarray(pt_true), np.asarray(pt_pred_m, dtype=np.float32)
                ),
                "majority_vote": compute_eval_metrics(
                    np.asarray(pt_true), np.asarray(pt_pred_v, dtype=np.float32)
                ),
            }

        # Output directory for this split
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Plots
        if len(np.unique(y_true_arr)) == 2:
            fpr, tpr, thresholds = compute_roc(y_true_arr, y_prob_arr)
            plot_roc_curve(
                fpr, tpr,
                auc=float(metrics_05.get("auc_roc", 0.0)),
                output_path=split_dir / "roc_curve.png",
                title=f"ROC Curve — {split}",
            )
            pr_prec, pr_rec, _ = compute_pr(y_true_arr, y_prob_arr)
            plot_pr_curve(
                pr_prec, pr_rec,
                avg_precision=float(metrics_05.get("avg_precision", 0.0)),
                output_path=split_dir / "pr_curve.png",
                title=f"Precision-Recall Curve — {split}",
            )

        # JSON report
        run_id = self.cfg.get("_run_id", "unknown")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = {
            "checkpoint": str(self.checkpoint_path.resolve()),
            "split": split,
            "run_id": run_id,
        }
        save_eval_json(
            per_roi,
            summary_metrics,
            output_path=split_dir / f"roi_eval_{split}_{ts}.json",
            metadata=metadata,
        )

        # Persistent eval CSV — written one level above run_dir (e.g. outputs/)
        # so it accumulates across all runs in one place.
        eval_history_path = self.output_dir.parent / "eval_history.csv"
        append_eval_summary_csv(
            summary_metrics,
            csv_path=eval_history_path,
            run_id=run_id,
            split=split,
        )

        auc = summary_metrics.get("auc_roc", float("nan"))
        f1 = metrics_05.get("f1", 0.0)
        log.info(
            "split=%s  auc_roc=%.4f  f1@0.5=%.4f  opt_threshold=%.4f",
            split, auc, f1, opt_threshold,
        )
        return summary_metrics

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_data_paths(
        self, data_cfg: dict[str, Any]
    ) -> tuple[Path, Path | None, Path | None]:
        def _res(raw: str) -> Path:
            p = Path(raw).expanduser()
            if p.is_absolute():
                return p.resolve()
            return (PROJECT_ROOT / p).resolve()

        manifest_path = _res(str(data_cfg.get("cube_manifest_csv", "")))
        mask_root_raw = data_cfg.get("mask_root")
        mask_root = _res(str(mask_root_raw)) if mask_root_raw else None
        if mask_root and not mask_root.exists():
            mask_root = None
        cube_root_raw = data_cfg.get("cube_root")
        cube_root = _res(str(cube_root_raw)) if cube_root_raw else None
        if cube_root and not cube_root.exists():
            cube_root = None
        return manifest_path, mask_root, cube_root

    def _resolve_cube_path(self, row: Any, cube_root: Path | None) -> Path:
        if cube_root and "patient_id" in row and "roi_name" in row:
            return cube_root / str(row["patient_id"]) / f"{row['roi_name']}.npy"
        return Path(str(row["output_path"]))

    def _resolve_mask_path(self, row: Any, mask_root: Path | None) -> Path | None:
        if not mask_root:
            return None
        if "patient_id" not in row or "roi_name" not in row:
            return None
        p = mask_root / str(row["patient_id"]) / f"{row['roi_name']}_mask.npy"
        return p if p.exists() else None
