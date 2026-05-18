"""K-fold cross-validation for 3D CNN models on HSI patches.

For each fold i:
  - test  = patients assigned to fold i
  - val   = patients assigned to fold (i+1) % k   (used for early stopping)
  - train = all remaining patients

Results are saved per fold in:
  outputs/kfold_{model}_{run_id}/fold_{i}/

After all folds a summary is printed and saved to:
  outputs/kfold_{model}_{run_id}/kfold_summary.json

Usage:
    python scripts/run_kfold.py --config configs/train/wavelet/baseline.yaml
    python scripts/run_kfold.py --config configs/train/wavelet/baseline.yaml --k 5
    python scripts/run_kfold.py --config configs/train/wavelet/baseline.yaml --folds 0,1,2  # subset
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import CubePatchDataset
from src.models import build_model_from_config
from src.training import (
    CheckpointCallback,
    EvalCallback,
    Trainer,
    build_optimizer,
    build_scheduler,
    compute_class_weights,
)
from src.utils.logging_setup import configure_logging, get_logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-fold CV for 3D CNN on HSI patches.")
    p.add_argument("--config", required=True, help="Train config YAML")
    p.add_argument("--model", default=None, help="Override model config YAML")
    p.add_argument("--kfold-manifest", default=None,
                   help="Manifest CSV with a 'fold' column (default: auto-detect from data.cube_manifest_csv stem + _kfold{k}.csv)")
    p.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    p.add_argument("--folds", default=None,
                   help="Comma-separated fold indices to run (default: all). E.g. --folds 0,1,2")
    p.add_argument("--no-auto-eval", action="store_true", help="Skip automatic eval after each fold")
    p.add_argument("--resume", default=None, metavar="KFOLD_RUN_DIR",
                   help="Path to an existing kfold run directory to resume. "
                        "Completed folds (with fold_result.json) are skipped automatically.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (same as run_train.py)
# ---------------------------------------------------------------------------

def _resolve_path(config_path: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    candidate = (config_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / p).resolve()


def _pick_device() -> torch.device:
    from src.utils.device import pick_training_device
    return pick_training_device()


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _save_fold_plots(report: dict, history: list[dict], fold_dir: Path, model_name: str, fold_idx: int) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(fold_dir / "history.csv", index=False)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train", color="C0")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", color="C1")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Fold {fold_idx} — Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["val_f1"] for h in history], color="C3")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1")
    axes[1].set_title(f"Fold {fold_idx} — Val F1"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fold_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_epoch = report.get("best_epoch", -1)
    best_row = next((h for h in history if h["epoch"] == best_epoch), history[-1] if history else None)
    summary_path = fold_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_idx}: {model_name}\n\n")
        f.write(f"Run ID: {report.get('run_id', 'N/A')}\n")
        f.write(f"Best epoch: {report.get('best_epoch', 'N/A')}\n")
        bl = report.get("best_val_loss")
        f.write(f"Best val loss: {bl:.4f}\n" if isinstance(bl, float) else "Best val loss: N/A\n")
        dur = report.get("duration_sec")
        f.write(f"Duration: {dur:.1f} s\n" if isinstance(dur, float) else "Duration: N/A s\n")
        if best_row:
            f.write(f"\nBest epoch val metrics:\n")
            f.write(f"  Accuracy:  {best_row.get('val_acc', 0):.4f}\n")
            f.write(f"  Precision: {best_row.get('val_precision', 0):.4f}\n")
            f.write(f"  Recall:    {best_row.get('val_recall', 0):.4f}\n")
            f.write(f"  F1:        {best_row.get('val_f1', 0):.4f}\n")


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def _train_fold(
    fold_idx: int,
    k: int,
    kfold_df: pd.DataFrame,
    cfg: dict,
    config_path: Path,
    model_config_path: Path,
    model_cfg: dict,
    kfold_run_dir: Path,
    run_id: str,
    device: torch.device,
    no_auto_eval: bool,
    log,
) -> dict:
    """Train one fold. Returns a result dict with fold metrics."""

    fold_dir = kfold_run_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip already-completed folds ──────────────────────────────────────
    fold_result_path = fold_dir / "fold_result.json"
    if fold_result_path.exists():
        log.info("Fold %d: already complete (fold_result.json found), skipping.", fold_idx)
        return json.loads(fold_result_path.read_text(encoding="utf-8"))

    # ── Split by fold column ──────────────────────────────────────────────
    test_fold = fold_idx
    val_fold = (fold_idx + 1) % k

    fold_df = kfold_df.copy()
    fold_df["split"] = "train"
    fold_df.loc[fold_df["fold"] == test_fold, "split"] = "test"
    fold_df.loc[fold_df["fold"] == val_fold, "split"] = "val"

    train_rows = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_rows   = fold_df[fold_df["split"] == "val"].reset_index(drop=True)

    test_patients  = sorted(fold_df[fold_df["split"] == "test"]["patient_id"].unique())
    val_patients   = sorted(fold_df[fold_df["split"] == "val"]["patient_id"].unique())
    train_patients = sorted(fold_df[fold_df["split"] == "train"]["patient_id"].unique())

    log.info("=== FOLD %d/%d ===", fold_idx, k - 1)
    log.info("  test  patients: %s", test_patients)
    log.info("  val   patients: %s", val_patients)
    log.info("  train patients: %s", train_patients)
    log.info("  train=%d  val=%d  test=%d ROIs",
             len(train_rows), len(val_rows),
             len(fold_df[fold_df["split"] == "test"]))

    # ── Datasets ─────────────────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    patch_h = int(data_cfg["patch_h"])
    patch_w = int(data_cfg["patch_w"])
    stride_h = int(data_cfg["stride_h"])
    stride_w = int(data_cfg["stride_w"])
    min_tissue = float(data_cfg.get("min_tissue_ratio", 0.0))
    val_seed = int(cfg.get("seed", 42))
    patches_per_cube = int(data_cfg.get("patches_per_cube", 1))
    use_all_patches = bool(data_cfg.get("use_all_patches", False))
    max_cached_cubes = int(data_cfg.get("max_cached_cubes", 12))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    augment = bool(data_cfg.get("augment", False))

    mask_root_raw = data_cfg.get("mask_root")
    mask_path = None
    if mask_root_raw:
        p = _resolve_path(config_path, mask_root_raw)
        mask_path = p if p.exists() else None

    cube_root_raw = data_cfg.get("cube_root")
    cube_root_path = None
    if cube_root_raw:
        p = _resolve_path(config_path, cube_root_raw)
        cube_root_path = p if p.exists() else None

    ds_kwargs = dict(
        patch_h=patch_h, patch_w=patch_w,
        mask_root=mask_path, min_tissue_ratio=min_tissue,
        cube_root=cube_root_path, patches_per_cube=patches_per_cube,
        stride_h=stride_h, stride_w=stride_w,
        use_all_patches=use_all_patches, max_cached_cubes=max_cached_cubes,
    )
    use_roi = str(cfg.get("trainer", {}).get("checkpoint_metric", "val_loss")) == "val_auc_roi"
    train_ds = CubePatchDataset(train_rows, val_seed=None, augment=augment, **ds_kwargs)
    val_ds   = CubePatchDataset(val_rows,   val_seed=val_seed, augment=False,
                                return_cube_idx=use_roi, **ds_kwargs)

    if len(train_ds) == 0:
        raise RuntimeError(f"Fold {fold_idx}: No train samples.")
    if len(val_ds) == 0:
        raise RuntimeError(f"Fold {fold_idx}: No val samples.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model_from_config(model_config_path).to(device)
    log.info("Model: %s  params=%d", model.__class__.__name__, _count_params(model))

    # ── Check for resume checkpoint ───────────────────────────────────────
    last_ckpt_path = fold_dir / "last.pt"
    resume_state: dict | None = None
    if last_ckpt_path.exists():
        log.info("Fold %d: found last.pt — resuming from checkpoint.", fold_idx)
        resume_state = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model_state_dict"])

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_cfg = cfg.get("loss", {})
    class_weighting = bool(loss_cfg.get("class_weighting", False))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if class_weighting:
        weights = compute_class_weights(train_rows["label_id"].tolist(), device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        log.info("Class weights: [%.4f, %.4f]", float(weights[0]), float(weights[1]))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model.parameters(), cfg.get("optimizer", {}))
    max_epochs = int(cfg.get("trainer", {}).get("max_epochs", 50))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), num_epochs=max_epochs)

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])

    trainer_cfg = cfg.get("trainer", {})
    amp_enabled = bool(trainer_cfg.get("mixed_precision", False)) and device.type == "cuda"

    # ── Callbacks ─────────────────────────────────────────────────────────
    fold_run_id = f"{run_id}_f{fold_idx}"
    ckpt_cb = CheckpointCallback(
        checkpoint_dir=fold_dir,
        model_name=model_config_path.stem,
        run_id=fold_run_id,
        model_config_path=model_config_path,
        train_config_path=config_path,
    )
    callbacks = [ckpt_cb]

    if not no_auto_eval:
        # Write the per-fold manifest to a temp CSV so EvalCallback can use it
        fold_manifest_path = fold_dir / "fold_manifest.csv"
        fold_df.to_csv(fold_manifest_path, index=False)
        eval_data_cfg = {
            **data_cfg,
            "cube_manifest_csv": str(fold_manifest_path),
            "cube_root": str(cube_root_path) if cube_root_path else data_cfg.get("cube_root"),
            "mask_root": str(mask_path) if mask_path else data_cfg.get("mask_root"),
        }
        eval_cb = EvalCallback(
            cfg={**cfg, "data": eval_data_cfg},
            output_dir=fold_dir,
            splits=["test"],   # val is used for early stopping; test is the held-out fold
            batch_size=batch_size,
        )
        callbacks.append(eval_cb)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=criterion,
        cfg=cfg,
        output_dir=fold_dir,
        run_id=fold_run_id,
        callbacks=callbacks,
        amp_enabled=amp_enabled,
        roi_val=use_roi,
    )
    final_logs = trainer.fit(resume_from=resume_state)

    _save_fold_plots(
        {**final_logs, "run_id": fold_run_id},
        final_logs["history"],
        fold_dir,
        model_config_path.stem,
        fold_idx,
    )

    # ── Collect test eval result ───────────────────────────────────────────
    test_eval_json = None
    if not no_auto_eval:
        test_jsons = sorted((fold_dir / "test").glob("roi_eval_test_*.json"))
        if test_jsons:
            test_eval_json = json.loads(test_jsons[-1].read_text(encoding="utf-8"))

    result = {
        "fold": fold_idx,
        "test_patients": test_patients,
        "val_patients": val_patients,
        "n_train_rois": len(train_rows),
        "n_val_rois": len(val_rows),
        "n_test_rois": len(fold_df[fold_df["split"] == "test"]),
        "best_epoch": final_logs.get("best_epoch"),
        "best_val_loss": final_logs.get("best_val_loss"),
        "duration_sec": final_logs.get("duration_sec"),
        "num_params": final_logs.get("num_params"),
        "test_eval": test_eval_json,
        "fold_dir": str(fold_dir),
    }
    fold_result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def _aggregate_results(fold_results: list[dict]) -> dict:
    auc_roc_list, ap_list, f1_list, acc_list = [], [], [], []
    for r in fold_results:
        te = r.get("test_eval")
        if te is None:
            continue
        roi_m = te.get("roi_metrics", {})
        auc_roc_list.append(roi_m.get("auc_roc", float("nan")))
        ap_list.append(roi_m.get("avg_precision", float("nan")))
        m05 = roi_m.get("metrics_at_threshold_0.5", {})
        f1_list.append(m05.get("f1", float("nan")))
        acc_list.append(m05.get("accuracy", float("nan")))

    def _stats(vals: list[float]) -> dict:
        arr = [v for v in vals if not np.isnan(v)]
        if not arr:
            return {"mean": None, "std": None, "min": None, "max": None, "values": vals}
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "values": vals,
        }

    return {
        "n_folds_with_eval": len(auc_roc_list),
        "auc_roc":      _stats(auc_roc_list),
        "avg_precision": _stats(ap_list),
        "f1_at_0.5":    _stats(f1_list),
        "accuracy_at_0.5": _stats(acc_list),
    }


def _print_summary(fold_results: list[dict], agg: dict) -> None:
    print("\n" + "=" * 72)
    print("K-FOLD SUMMARY")
    print("=" * 72)
    header = f"{'Fold':<6} {'Test patients':<26} {'AUC-ROC':<10} {'AvgPrec':<10} {'F1@0.5':<10} {'Acc@0.5':<10}"
    print(header)
    print("-" * 72)
    for r in fold_results:
        te = r.get("test_eval")
        if te is None:
            print(f"{r['fold']:<6} {','.join(r['test_patients']):<26} {'N/A (no eval)'}")
            continue
        roi_m = te.get("roi_metrics", {})
        m05 = roi_m.get("metrics_at_threshold_0.5", {})
        auc  = roi_m.get("auc_roc", float("nan"))
        ap   = roi_m.get("avg_precision", float("nan"))
        f1   = m05.get("f1", float("nan"))
        acc  = m05.get("accuracy", float("nan"))
        print(f"{r['fold']:<6} {','.join(r['test_patients']):<26} {auc:<10.4f} {ap:<10.4f} {f1:<10.4f} {acc:<10.4f}")
    print("-" * 72)
    for metric, label in [("auc_roc", "AUC-ROC"), ("avg_precision", "AvgPrec"),
                           ("f1_at_0.5", "F1@0.5"), ("accuracy_at_0.5", "Acc@0.5")]:
        s = agg[metric]
        if s["mean"] is not None:
            print(f"  {label:<18} mean={s['mean']:.4f}  std={s['std']:.4f}  [{s['min']:.4f} – {s['max']:.4f}]")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_cfg_path_raw = args.model or cfg.get("model_config", "configs/models/baseline_3dcnn.yaml")
    model_config_path = _resolve_path(config_path, model_cfg_path_raw)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_cfg: dict[str, Any] = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))

    data_cfg = cfg.get("data", {})
    for key in ("patch_h", "patch_w", "stride_h", "stride_w"):
        if key not in data_cfg:
            raise ValueError(f"data.{key} missing in {config_path}")

    k = args.k

    # Resolve kfold manifest
    if args.kfold_manifest:
        kfold_manifest_path = _resolve_path(config_path, args.kfold_manifest)
    else:
        base_manifest_raw = data_cfg.get("cube_manifest_csv", "data/interim/wavelet_manifest.csv")
        base_manifest = _resolve_path(config_path, base_manifest_raw)
        kfold_manifest_path = base_manifest.parent / f"{base_manifest.stem}_kfold{k}.csv"

    if not kfold_manifest_path.exists():
        raise FileNotFoundError(
            f"K-fold manifest not found: {kfold_manifest_path}\n"
            f"Run: python scripts/preprocessing/split/make_kfold_split.py --k {k}"
        )

    kfold_df = pd.read_csv(kfold_manifest_path)
    if "fold" not in kfold_df.columns:
        raise ValueError(f"Manifest {kfold_manifest_path} has no 'fold' column. Re-run make_kfold_split.py.")

    # Which folds to run
    folds_to_run: list[int]
    if args.folds:
        folds_to_run = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds_to_run = list(range(k))

    model_name = model_config_path.stem
    paths_cfg = cfg.get("paths", {})
    outputs_base = _resolve_path(config_path, str(paths_cfg.get("outputs_dir", "outputs")))

    if args.resume:
        kfold_run_dir = Path(args.resume).expanduser().resolve()
        if not kfold_run_dir.exists():
            kfold_run_dir = (PROJECT_ROOT / args.resume).resolve()
        if not kfold_run_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {args.resume}")
        # Recover run_id from directory name (kfold_<model>_<run_id>)
        run_id = kfold_run_dir.name.split("_", 2)[-1] if "_" in kfold_run_dir.name else kfold_run_dir.name
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        kfold_run_dir = outputs_base / f"kfold_{model_name}_{run_id}"
        kfold_run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir=kfold_run_dir, run_name="kfold")
    log = get_logger(__name__)

    device = _pick_device()
    log.info("device=%s  model=%s  k=%d  run_id=%s", device.type, model_name, k, run_id)
    log.info("kfold_manifest=%s", kfold_manifest_path)
    log.info("folds: %s", folds_to_run)
    log.info("kfold_run_dir=%s", kfold_run_dir)

    fold_results: list[dict] = []
    total_start = time.time()

    for fold_idx in folds_to_run:
        result = _train_fold(
            fold_idx=fold_idx,
            k=k,
            kfold_df=kfold_df,
            cfg=deepcopy(cfg),
            config_path=config_path,
            model_config_path=model_config_path,
            model_cfg=model_cfg,
            kfold_run_dir=kfold_run_dir,
            run_id=run_id,
            device=device,
            no_auto_eval=args.no_auto_eval,
            log=log,
        )
        fold_results.append(result)
        if result.get("test_eval"):
            auc = result["test_eval"].get("roi_metrics", {}).get("auc_roc", float("nan"))
            f1  = result["test_eval"].get("roi_metrics", {}).get("metrics_at_threshold_0.5", {}).get("f1", float("nan"))
            log.info("Fold %d done: test AUC-ROC=%.4f  F1=%.4f  (%.0f s)",
                     fold_idx, auc, f1, result.get("duration_sec", 0))
        else:
            log.info("Fold %d done (no eval). (%.0f s)", fold_idx, result.get("duration_sec", 0))

    total_dur = time.time() - total_start
    log.info("All folds done in %.0f s (%.1f min)", total_dur, total_dur / 60)

    agg = _aggregate_results(fold_results)
    _print_summary(fold_results, agg)

    # Save summary JSON
    summary = {
        "run_id": run_id,
        "k": k,
        "folds_run": folds_to_run,
        "model": model_name,
        "config": args.config,
        "kfold_manifest": str(kfold_manifest_path),
        "device": device.type,
        "total_duration_sec": total_dur,
        "aggregate": agg,
        "folds": [
            {k_: v for k_, v in r.items() if k_ != "test_eval"}
            for r in fold_results
        ],
    }
    summary_path = kfold_run_dir / "kfold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Save per-fold test eval JSONs inline for convenience
    full_summary_path = kfold_run_dir / "kfold_full_summary.json"
    full_summary_path.write_text(
        json.dumps({**summary, "folds": fold_results}, indent=2, default=str),
        encoding="utf-8"
    )

    log.info("Summary saved to: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
