"""Train 3D CNN models on HSI patches."""

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
from tqdm import tqdm
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D CNN on patches (on-the-fly or pre-built).")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Train config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override model config YAML")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest CSV (patch or cube)")
    parser.add_argument("--cube-manifest", type=str, default=None, help="Use cube manifest for on-the-fly patching")
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


def _binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = torch.argmax(logits, dim=1)
    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / total

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
    epoch: int | None = None,
    max_epochs: int | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    desc = f"Epoch {epoch}/{max_epochs} train" if (epoch and max_epochs) else "train"
    it = tqdm(loader, desc=desc, unit="batch", leave=False) if (epoch and max_epochs) else loader

    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_n = int(y.size(0))
        running_loss += float(loss.detach().item()) * batch_n
        total += batch_n
        if hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{running_loss / total:.4f}")

    return running_loss / max(1, total)


@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    max_epochs: int | None = None,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    desc = f"Epoch {epoch}/{max_epochs} val" if (epoch and max_epochs) else "val"
    it = tqdm(loader, desc=desc, unit="batch", leave=False) if (epoch and max_epochs) else loader

    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        batch_n = int(y.size(0))
        running_loss += float(loss.detach().item()) * batch_n
        total += batch_n

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    if total == 0:
        return {"loss": float("inf"), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = _binary_metrics(logits_cat, targets_cat)
    metrics["loss"] = running_loss / total
    return metrics


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _to_yaml_friendly(obj: Any) -> Any:
    """Gjør nested strukturer om til typer PyYAML safe_dump kan serialisere (unngår Version, numpy, osv.)."""
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
    """Siste sikring: alt som ikke er JSON-vennlig blir til streng (f.eks. packaging.version.Version)."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError, OverflowError):
        return _to_yaml_friendly(obj)


def _try_git_short_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _save_hyperparams_snapshot(
    run_dir: Path,
    *,
    run_id: str,
    config_path: Path,
    model_config_path: Path,
    manifest_path: Path,
    mask_path: Path | None,
    cube_root_path: Path | None,
    cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    args: argparse.Namespace,
    lr: float,
    weight_decay: float,
    class_weighting: bool,
    class_weights: list[float] | None,
    use_scheduler: bool,
    t_max_epochs: int | None,
    max_epochs: int,
    patience: int,
    amp_requested: bool,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Path:
    """Skriv hyperparams.yaml i run_dir med alt som ble brukt i denne kjøringen."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "hyperparams.yaml"

    train_yaml_source = config_path.read_text(encoding="utf-8")
    model_yaml_source = model_config_path.read_text(encoding="utf-8")

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
        "effective_training": {
            "optimizer": "Adam",
            "lr": lr,
            "weight_decay": weight_decay,
            "class_weighting": class_weighting,
            "class_weights": class_weights,
            "scheduler_cosine": use_scheduler,
            "scheduler_t_max_epochs": t_max_epochs,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "mixed_precision_requested": amp_requested,
            "mixed_precision_active": amp_enabled,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "train_config_parsed": cfg,
        "model_config_parsed": model_cfg,
        "train_config_source_yaml": train_yaml_source,
        "model_config_source_yaml": model_yaml_source,
    }

    snapshot_out = _snapshot_fully_json_safe(_to_yaml_friendly(snapshot))
    try:
        text = yaml.safe_dump(
            snapshot_out,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=120,
        )
        out_path.write_text(text, encoding="utf-8")
    except yaml.representer.RepresenterError:
        # Sjelden: typer PyYAML fortsatt ikke tåler — skriv JSON med samme innhold
        json_path = run_dir / "hyperparams.json"
        json_path.write_text(json.dumps(snapshot_out, indent=2, ensure_ascii=False), encoding="utf-8")
        out_path = json_path
    return out_path


def _save_training_logs(
    report: dict,
    history: list[dict],
    run_dir: Path,
    model_name: str,
) -> None:
    """Lagre CSV, grafer og confusion matrix for sluttrapport."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. History som CSV
    df = pd.DataFrame(history)
    csv_path = run_dir / "history.csv"
    df.to_csv(csv_path, index=False)

    # 2. Treningskurver (loss, accuracy, F1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    epochs = [h["epoch"] for h in history]

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train", color="C0")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", color="C1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_acc"] for h in history], color="C2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [h["val_f1"] for h in history], color="C3")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].set_title("Validation F1")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Confusion matrix fra beste epoke
    best_epoch = report.get("best_epoch", -1)
    best_row = next((h for h in history if h["epoch"] == best_epoch), history[-1] if history else None)
    if best_row and all(k in best_row for k in ["tp", "tn", "fp", "fn"]):
        cm = np.array([[best_row["tn"], best_row["fp"]], [best_row["fn"], best_row["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        plt.colorbar(im, ax=ax, label="Count")
        ax.set_title(f"Confusion Matrix (epoch {best_epoch})")
        plt.tight_layout()
        plt.savefig(run_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Kort sammendrag for rapport
    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Treningsrapport: {model_name}\n\n")
        f.write(f"Run ID: {report.get('run_id', 'N/A')}\n")
        f.write(f"Device: {report.get('device', 'N/A')}\n")
        f.write(f"Best epoch: {report.get('best_epoch', 'N/A')}\n")
        bl = report.get("best_val_loss")
        f.write(f"Best val loss: {bl:.4f}\n" if isinstance(bl, (int, float)) else f"Best val loss: N/A\n")
        f.write(f"Train samples: {report.get('train_samples', 'N/A')}\n")
        f.write(f"Val samples: {report.get('val_samples', 'N/A')}\n")
        np_ = report.get("num_params")
        f.write(f"Params: {np_:,}\n" if isinstance(np_, int) else f"Params: N/A\n")
        dur = report.get("duration_sec")
        f.write(f"Duration: {dur:.1f} s\n" if isinstance(dur, (int, float)) else f"Duration: N/A\n")
        if best_row:
            f.write(f"\nBest epoch metrics:\n")
            f.write(f"  Accuracy:  {best_row.get('val_acc', 0):.4f}\n")
            f.write(f"  Precision: {best_row.get('val_precision', 0):.4f}\n")
            f.write(f"  Recall:    {best_row.get('val_recall', 0):.4f}\n")
            f.write(f"  F1:        {best_row.get('val_f1', 0):.4f}\n")


def main() -> int:
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {args.config}")

    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_cfg_path = args.model or cfg.get("model_config", "configs/models/baseline_3dcnn.yaml")
    model_config_path = _resolve_path(config_path, model_cfg_path)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_cfg: dict[str, Any] = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))

    data_cfg = cfg.get("data", {})
    _geom = ("patch_h", "patch_w", "stride_h", "stride_w")
    _missing = [k for k in _geom if k not in data_cfg]
    if _missing:
        raise ValueError(
            f"Set {', '.join('data.' + k for k in _geom)} in {config_path} (no defaults in code). "
            f"Missing: {_missing}"
        )
    patch_h = int(data_cfg["patch_h"])
    patch_w = int(data_cfg["patch_w"])
    stride_h = int(data_cfg["stride_h"])
    stride_w = int(data_cfg["stride_w"])
    manifest_override = (
        args.manifest
        or args.cube_manifest
        or data_cfg.get("cube_manifest_csv")
    )

    if not manifest_override:
        raise ValueError(
            "Missing manifest. Set data.cube_manifest_csv in configs/train.yaml, or pass --manifest / --cube-manifest"
        )

    manifest_path = _resolve_path(config_path, manifest_override)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_df = pd.read_csv(manifest_path)
    required = {"output_path", "label_id", "split"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Cube manifest missing columns: {sorted(missing)}")

    if "patient_id" not in train_df.columns or "roi_name" not in train_df.columns:
        raise ValueError("Cube manifest needs patient_id and roi_name for mask lookup")

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

    train_rows = train_df[train_df["split"] == "train"].reset_index(drop=True)
    val_rows = train_df[train_df["split"] == "val"].reset_index(drop=True)

    print(f"[train] patch size: {patch_h}x{patch_w}")
    print(f"[train] patches per cube: {'all' if use_all_patches else patches_per_cube}")
    print(f"[train] max_cached_cubes: {max_cached_cubes}")

    train_ds = CubePatchDataset(
        train_rows,
        patch_h=patch_h,
        patch_w=patch_w,
        mask_root=mask_path,
        min_tissue_ratio=min_tissue,
        val_seed=None,
        cube_root=cube_root_path,
        patches_per_cube=patches_per_cube,
        stride_h=stride_h,
        stride_w=stride_w,
        use_all_patches=use_all_patches,
        max_cached_cubes=max_cached_cubes,
    )
    val_ds = CubePatchDataset(
        val_rows,
        patch_h=patch_h,
        patch_w=patch_w,
        mask_root=mask_path,
        min_tissue_ratio=min_tissue,
        val_seed=val_seed,
        cube_root=cube_root_path,
        patches_per_cube=patches_per_cube,
        stride_h=stride_h,
        stride_w=stride_w,
        use_all_patches=use_all_patches,
        max_cached_cubes=max_cached_cubes,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No train samples.")
    if len(val_ds) == 0:
        raise RuntimeError("No val samples.")

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))

    device = _pick_device()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model_from_config(model_config_path).to(device)

    loss_cfg = cfg.get("loss", {})
    class_weighting = bool(loss_cfg.get("class_weighting", False))
    class_weights_list: list[float] | None = None
    if class_weighting:
        counts = train_rows["label_id"].value_counts().to_dict()
        n0 = float(counts.get(0, 1.0))
        n1 = float(counts.get(1, 1.0))
        total = n0 + n1
        weights = torch.tensor([total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float32, device=device)
        class_weights_list = [float(weights[0].item()), float(weights[1].item())]
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1.0e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1.0e-4))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sch_cfg = cfg.get("scheduler", {})
    use_scheduler = bool(sch_cfg.get("enabled", False)) and str(sch_cfg.get("name", "")).lower() == "cosine"
    scheduler = None
    t_max_epochs: int | None = None
    if use_scheduler:
        t_max = int(sch_cfg.get("t_max_epochs", cfg.get("trainer", {}).get("max_epochs", 50)))
        t_max_epochs = max(1, t_max)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max_epochs)

    trainer_cfg = cfg.get("trainer", {})
    max_epochs = int(trainer_cfg.get("max_epochs", 50))
    patience = int(trainer_cfg.get("early_stopping_patience", 10))
    amp_requested = bool(trainer_cfg.get("mixed_precision", False))
    amp_enabled = amp_requested and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    paths_cfg = cfg.get("paths", {})
    checkpoints_dir = _resolve_path(config_path, str(paths_cfg.get("checkpoints_dir", "outputs/checkpoints")))
    reports_dir = _resolve_path(config_path, str(paths_cfg.get("reports_dir", "outputs/reports")))
    plots_dir = _resolve_path(config_path, str(paths_cfg.get("plots_dir", "outputs/plots")))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = model_config_path.stem
    best_ckpt_path = checkpoints_dir / f"{model_name}_{run_id}_best.pt"
    last_ckpt_path = checkpoints_dir / f"{model_name}_{run_id}_last.pt"

    print(f"[train] device={device.type} model={model_name}")
    print(f"[train] manifest={manifest_path}")
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    t0 = time.perf_counter()
    epoch_iter = tqdm(range(1, max_epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_iter:
        epoch_iter.set_postfix(epoch=epoch, best_val=f"{best_val_loss:.4f}")
        train_loss = _run_train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_enabled,
            epoch=epoch, max_epochs=max_epochs,
        )
        val_metrics = _run_eval_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, max_epochs=max_epochs,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "tp": val_metrics.get("tp", 0),
            "tn": val_metrics.get("tn", 0),
            "fp": val_metrics.get("fp", 0),
            "fn": val_metrics.get("fn", 0),
        }
        history.append(row)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f}"
        )

        improved = row["val_loss"] < best_val_loss
        if improved:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config_path": str(model_config_path),
                    "train_config_path": str(config_path),
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config_path": str(model_config_path),
                "train_config_path": str(config_path),
                "val_metrics": val_metrics,
            },
            last_ckpt_path,
        )

        if epochs_no_improve >= patience:
            print(f"[train] early stopping at epoch {epoch} (patience={patience})")
            break

    duration_sec = time.perf_counter() - t0

    report = {
        "run_id": run_id,
        "device": device.type,
        "model_config_path": str(model_config_path),
        "train_config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history": history,
        "num_params": _count_params(model),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "duration_sec": duration_sec,
    }

    report_path = reports_dir / f"train_report_{model_name}_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    run_plots_dir = plots_dir / f"{model_name}_{run_id}"
    hp_path = _save_hyperparams_snapshot(
        run_plots_dir,
        run_id=run_id,
        config_path=config_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        mask_path=mask_path,
        cube_root_path=cube_root_path,
        cfg=cfg,
        model_cfg=model_cfg,
        args=args,
        lr=lr,
        weight_decay=weight_decay,
        class_weighting=class_weighting,
        class_weights=class_weights_list,
        use_scheduler=use_scheduler,
        t_max_epochs=t_max_epochs,
        max_epochs=max_epochs,
        patience=patience,
        amp_requested=amp_requested,
        amp_enabled=amp_enabled,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    _save_training_logs(report, history, run_plots_dir, model_name)

    print(f"[train] best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}")
    print(f"[train] best_ckpt={best_ckpt_path}")
    print(f"[train] report={report_path}")
    print(f"[train] plots={run_plots_dir} (hyperparams.yaml, history.csv, training_curves.png, confusion_matrix.png, summary.txt)")
    print(f"[train] hyperparams={hp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
