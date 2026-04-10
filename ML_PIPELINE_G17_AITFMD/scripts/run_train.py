"""Train 3D CNN models on HSI patches.

Thin entry-point script. All logic lives in src/training/.
Auto-eval (val + test) runs automatically after training via EvalCallback.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D CNN on patches (on-the-fly or pre-built).")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Train config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model config YAML")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest CSV (patch or cube)")
    parser.add_argument("--cube-manifest", type=str, default=None, help="Use cube manifest for on-the-fly patching")
    parser.add_argument("--no-auto-eval", action="store_true", help="Skip automatic eval after training")
    parser.add_argument("--eval-splits", type=str, default="val,test", help="Comma-separated splits for auto-eval")
    return parser.parse_args()


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


def _try_git_short_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _to_yaml_friendly(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {str(k): _to_yaml_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_friendly(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_to_yaml_friendly(v) for v in obj]
    try:
        if isinstance(obj, np.ndarray):
            return _to_yaml_friendly(obj.tolist())
        if isinstance(obj, np.generic):
            return _to_yaml_friendly(obj.item())
    except TypeError:
        pass
    return str(obj)


def _snapshot_fully_json_safe(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError, OverflowError):
        return _to_yaml_friendly(obj)


def _derive_dataset_name(data_cfg: dict, manifest_path: Path, cube_root_raw: str | None) -> str:
    explicit = str(data_cfg.get("dataset_name", "")).strip()
    if explicit:
        return explicit
    if cube_root_raw:
        name = Path(cube_root_raw).name
        if name:
            return name
    stem = manifest_path.stem
    return stem.replace("_manifest", "").replace("_baseline", "") or "unknown"


def _save_hyperparams_snapshot(run_dir: Path, *, run_id: str, config_path: Path,
                                model_config_path: Path, manifest_path: Path,
                                mask_path: Path | None, cube_root_path: Path | None,
                                cfg: dict, model_cfg: dict, args: argparse.Namespace,
                                device: torch.device) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "hyperparams.yaml"

    snapshot: dict[str, Any] = {
        "run_id": run_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": device.type,
            "git_short_sha": _try_git_short_sha(PROJECT_ROOT),
        },
        "cli": {
            "argv": sys.argv,
            "config": args.config,
            "model": args.model,
            "manifest": args.manifest,
            "cube_manifest": args.cube_manifest,
        },
        "resolved_paths": {
            "train_config": str(config_path.resolve()),
            "model_config": str(model_config_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "mask_root": str(mask_path.resolve()) if mask_path else None,
            "cube_root": str(cube_root_path.resolve()) if cube_root_path else None,
        },
        "train_config_parsed": cfg,
        "model_config_parsed": model_cfg,
        "train_config_source_yaml": config_path.read_text(encoding="utf-8"),
        "model_config_source_yaml": model_config_path.read_text(encoding="utf-8"),
    }

    snapshot_out = _snapshot_fully_json_safe(_to_yaml_friendly(snapshot))
    try:
        text = yaml.safe_dump(snapshot_out, sort_keys=False, allow_unicode=True,
                              default_flow_style=False, width=120)
        out_path.write_text(text, encoding="utf-8")
    except yaml.representer.RepresenterError:
        json_path = run_dir / "hyperparams.json"
        json_path.write_text(json.dumps(snapshot_out, indent=2, ensure_ascii=False), encoding="utf-8")
        out_path = json_path
    return out_path


def _save_training_plots(report: dict, history: list[dict], run_dir: Path, model_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(run_dir / "history.csv", index=False)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train", color="C0")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", color="C1")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["val_acc"] for h in history], color="C2")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Validation Accuracy")
    axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, [h["val_f1"] for h in history], color="C3")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1"); axes[2].set_title("Validation F1")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_epoch = report.get("best_epoch", -1)
    best_row = next((h for h in history if h["epoch"] == best_epoch), history[-1] if history else None)
    if best_row and all(k in best_row for k in ["tp", "tn", "fp", "fn"]):
        cm = np.array([[best_row["tn"], best_row["fp"]], [best_row["fn"], best_row["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"]); ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        plt.colorbar(im, ax=ax, label="Count")
        ax.set_title(f"Confusion Matrix (epoch {best_epoch})")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Treningsrapport: {model_name}\n\n")
        f.write(f"Run ID: {report.get('run_id', 'N/A')}\n")
        f.write(f"Device: {report.get('device', 'N/A')}\n")
        f.write(f"Best epoch: {report.get('best_epoch', 'N/A')}\n")
        bl = report.get("best_val_loss")
        f.write(f"Best val loss: {bl:.4f}\n" if isinstance(bl, (int, float)) else "Best val loss: N/A\n")
        np_ = report.get("num_params")
        f.write(f"Params: {np_:,}\n" if isinstance(np_, int) else "Params: N/A\n")
        dur = report.get("duration_sec")
        f.write(f"Duration: {dur:.1f} s\n" if isinstance(dur, (int, float)) else "Duration: N/A\n")
        if best_row:
            f.write(f"\nBest epoch metrics:\n")
            f.write(f"  Accuracy:  {best_row.get('val_acc', 0):.4f}\n")
            f.write(f"  Precision: {best_row.get('val_precision', 0):.4f}\n")
            f.write(f"  Recall:    {best_row.get('val_recall', 0):.4f}\n")
            f.write(f"  F1:        {best_row.get('val_f1', 0):.4f}\n")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _print_model_summary(model: nn.Module, model_cfg: dict, model_config_path: Path, log) -> None:
    log.info("--- model / architecture ---")
    log.info("model_config: %s", model_config_path.resolve())
    arch = model_cfg.get("model", {}).get("architecture") or model_cfg.get("architecture")
    if isinstance(arch, dict) and arch:
        keys = ("in_channels", "num_classes", "channels", "kernel_size", "dropout", "base_channels", "num_blocks")
        bits = [f"{k}={arch[k]!r}" for k in keys if k in arch]
        if bits:
            log.info("YAML architecture: %s", ", ".join(bits))
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Instance: %s | params: %d (trainable: %d)", model.__class__.__name__, n_params, n_train)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            log.debug("  Conv3d %s: in=%d out=%d kernel=%s", name, m.in_channels, m.out_channels, tuple(m.kernel_size))
        elif isinstance(m, nn.MaxPool3d):
            log.debug("  MaxPool3d %s: kernel=%s stride=%s", name, tuple(m.kernel_size), tuple(m.stride))
        elif isinstance(m, nn.Linear):
            log.debug("  Linear %s: in=%d out=%d", name, m.in_features, m.out_features)
    log.info("--- (end architecture) ---")


def main() -> int:
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {args.config}")

    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths_cfg = cfg.get("paths", {})
    log = get_logger(__name__)

    model_cfg_path_raw = args.model or cfg.get("model_config", "configs/models/baseline_3dcnn.yaml")
    model_config_path = _resolve_path(config_path, model_cfg_path_raw)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_cfg: dict[str, Any] = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))

    data_cfg = cfg.get("data", {})
    for k in ("patch_h", "patch_w", "stride_h", "stride_w"):
        if k not in data_cfg:
            raise ValueError(f"data.{k} missing in {config_path}")

    patch_h = int(data_cfg["patch_h"])
    patch_w = int(data_cfg["patch_w"])
    stride_h = int(data_cfg["stride_h"])
    stride_w = int(data_cfg["stride_w"])
    manifest_override = args.manifest or args.cube_manifest or data_cfg.get("cube_manifest_csv")
    if not manifest_override:
        raise ValueError("Missing manifest. Set data.cube_manifest_csv or pass --manifest.")

    manifest_path = _resolve_path(config_path, manifest_override)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_df = pd.read_csv(manifest_path)
    required = {"output_path", "label_id", "split", "patient_id", "roi_name"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Cube manifest missing columns: {sorted(missing)}")

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

    min_tissue = float(data_cfg.get("min_tissue_ratio", 0.0))
    val_seed = int(cfg.get("seed", 42))
    patches_per_cube = int(data_cfg.get("patches_per_cube", 1))
    use_all_patches = bool(data_cfg.get("use_all_patches", False))
    max_cached_cubes = int(data_cfg.get("max_cached_cubes", 12))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_rows = train_df[train_df["split"] == "train"].reset_index(drop=True)
    val_rows = train_df[train_df["split"] == "val"].reset_index(drop=True)

    log.info("patch size: %dx%d  patches_per_cube: %s  max_cached_cubes: %d",
             patch_h, patch_w, "all" if use_all_patches else patches_per_cube, max_cached_cubes)

    augment = bool(data_cfg.get("augment", False))
    if augment:
        log.info("Spatial augmentation enabled (hflip, vflip, rot90)")

    ds_kwargs = dict(
        patch_h=patch_h, patch_w=patch_w,
        mask_root=mask_path, min_tissue_ratio=min_tissue,
        cube_root=cube_root_path, patches_per_cube=patches_per_cube,
        stride_h=stride_h, stride_w=stride_w,
        use_all_patches=use_all_patches, max_cached_cubes=max_cached_cubes,
    )
    train_ds = CubePatchDataset(train_rows, val_seed=None, augment=augment, **ds_kwargs)
    val_ds = CubePatchDataset(val_rows, val_seed=val_seed, augment=False, **ds_kwargs)

    if len(train_ds) == 0:
        raise RuntimeError("No train samples.")
    if len(val_ds) == 0:
        raise RuntimeError("No val samples.")

    device = _pick_device()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = build_model_from_config(model_config_path).to(device)
    _print_model_summary(model, model_cfg, model_config_path, log)

    # Loss
    loss_cfg = cfg.get("loss", {})
    class_weighting = bool(loss_cfg.get("class_weighting", False))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if class_weighting:
        weights = compute_class_weights(train_rows["label_id"].tolist(), device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        log.info("Class weights: [%.4f, %.4f]  label_smoothing=%.2f",
                 float(weights[0]), float(weights[1]), label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            log.info("label_smoothing=%.2f", label_smoothing)

    # Optimizer + scheduler
    optimizer = build_optimizer(model.parameters(), cfg.get("optimizer", {}))
    max_epochs = int(cfg.get("trainer", {}).get("max_epochs", 50))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), num_epochs=max_epochs)

    trainer_cfg = cfg.get("trainer", {})
    amp_requested = bool(trainer_cfg.get("mixed_precision", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # All outputs land in one folder per run: outputs/{model_name}_{run_id}/
    model_name = model_config_path.stem
    outputs_base = _resolve_path(config_path, str(paths_cfg.get("outputs_dir", "outputs")))
    run_dir = outputs_base / f"{model_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_dir=run_dir, run_name="run")
    log = get_logger(__name__)

    log.info("device=%s  model=%s  run_id=%s", device.type, model_name, run_id)
    log.info("run_dir=%s", run_dir)
    log.info("train=%d  val=%d  manifest=%s", len(train_ds), len(val_ds), manifest_path)

    # Sanity-check forward pass
    try:
        x0, _ = train_ds[0]
        with torch.no_grad():
            model(x0.unsqueeze(0).to(device))
    except RuntimeError as e:
        log.error("Forward pass failed (patch too small for network?): %s", e)
        raise SystemExit(1) from e

    # Callbacks
    ckpt_cb = CheckpointCallback(
        checkpoint_dir=run_dir,
        model_name=model_name,
        run_id=run_id,
        model_config_path=model_config_path,
        train_config_path=config_path,
    )
    callbacks = [ckpt_cb]

    if not args.no_auto_eval:
        eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
        eval_data_cfg = {
            **data_cfg,
            "cube_manifest_csv": str(manifest_path),
            "cube_root": str(cube_root_path) if cube_root_path else data_cfg.get("cube_root"),
            "mask_root": str(mask_path) if mask_path else data_cfg.get("mask_root"),
        }
        eval_cb = EvalCallback(
            cfg={**cfg, "data": eval_data_cfg},
            output_dir=run_dir,
            splits=eval_splits,
            batch_size=batch_size,
        )
        callbacks.append(eval_cb)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=criterion,
        cfg=cfg,
        output_dir=run_dir,
        run_id=run_id,
        callbacks=callbacks,
        amp_enabled=amp_enabled,
    )

    final_logs = trainer.fit()

    # Save training artefacts — all into run_dir
    hp_path = _save_hyperparams_snapshot(
        run_dir,
        run_id=run_id, config_path=config_path, model_config_path=model_config_path,
        manifest_path=manifest_path, mask_path=mask_path, cube_root_path=cube_root_path,
        cfg=cfg, model_cfg=model_cfg, args=args, device=device,
    )
    _save_training_plots(
        {**final_logs, "run_id": run_id, "device": device.type},
        final_logs["history"],
        run_dir,
        model_name,
    )

    report = {
        "run_id": run_id,
        "device": device.type,
        "model_config_path": str(model_config_path),
        "train_config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "best_epoch": final_logs["best_epoch"],
        "best_val_loss": final_logs["best_val_loss"],
        "best_checkpoint": str(trainer.best_checkpoint_path) if trainer.best_checkpoint_path else None,
        "history": final_logs["history"],
        "num_params": final_logs["num_params"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "duration_sec": final_logs["duration_sec"],
    }
    report_path = run_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    log.info("best_epoch=%d  best_val_loss=%.4f", final_logs["best_epoch"], final_logs["best_val_loss"])
    log.info("All outputs in: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
