"""Trainer class: owns the training loop and fires callback hooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import BaseCallback
from src.training.early_stopping import EarlyStopping
from src.training.metrics import compute_binary_metrics, compute_roi_metrics, format_metrics_table
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class Trainer:
    """Encapsulates the full training loop for a 3D CNN classifier.

    The Trainer is responsible for the forward/backward pass, metric
    collection, early stopping, AMP scaler, and firing callback hooks.
    Everything else (checkpointing, eval, logging setup) lives in callbacks.

    Args:
        model: Untrained ``nn.Module`` already moved to ``device``.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Configured optimizer (from ``src.training.optim``).
        scheduler: LR scheduler, or ``None``.
        loss_fn: Loss criterion (e.g. ``nn.CrossEntropyLoss``).
        cfg: Full parsed ``train.yaml`` dict.
        output_dir: Base output directory for this run.
        run_id: Unique run identifier (e.g. ``"20250410_120000"``).
        callbacks: List of ``BaseCallback`` instances.
        amp_enabled: Whether to use Automatic Mixed Precision.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        cfg: dict[str, Any],
        output_dir: Path,
        run_id: str,
        callbacks: list[BaseCallback] | None = None,
        amp_enabled: bool = False,
        roi_val: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.callbacks: list[BaseCallback] = callbacks or []
        self.amp_enabled = amp_enabled
        self.roi_val = roi_val

        trainer_cfg = cfg.get("trainer", {})
        self.max_epochs = int(trainer_cfg.get("max_epochs", 50))
        patience = int(trainer_cfg.get("early_stopping_patience", 10))
        min_delta = float(trainer_cfg.get("early_stopping_min_delta", 0.0))
        self.checkpoint_metric = str(trainer_cfg.get("checkpoint_metric", "val_loss"))
        es_mode = "max" if self.checkpoint_metric == "val_auc_roi" else "min"
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode=es_mode)

        device_type = next(model.parameters()).device.type
        self.device = next(model.parameters()).device
        self.scaler = GradScaler(device=device_type, enabled=amp_enabled)

        self.history: list[dict[str, Any]] = []
        self.best_checkpoint_path: Path | None = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1
        self.best_auc_roi: float = float("-inf")
        self.best_f1_roi_opt: float = float("-inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int | None = None, resume_from: dict | None = None) -> dict[str, Any]:
        """Run the training loop.

        Args:
            num_epochs: Override ``trainer.max_epochs`` from config.
            resume_from: Checkpoint dict loaded from ``last.pt``. If given,
                restores training state (best_val_loss, best_epoch, history,
                scheduler) and starts from the next epoch.

        Returns:
            Final ``logs`` dict (also accessible via ``self.history``).
        """
        max_epochs = num_epochs or self.max_epochs
        start_epoch = 1

        if resume_from is not None:
            start_epoch = int(resume_from["epoch"]) + 1
            self.best_val_loss = float(resume_from.get("best_val_loss", float("inf")))
            self.best_epoch = int(resume_from.get("best_epoch", -1))
            self.best_auc_roi = float(resume_from.get("best_auc_roi", float("-inf")))
            self.best_f1_roi_opt = float(resume_from.get("best_f1_roi_opt", float("-inf")))
            self.history = list(resume_from.get("history", []))
            sched_state = resume_from.get("scheduler_state_dict")
            if sched_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(sched_state)
            # Restore best_checkpoint_path so EvalCallback can find the best model
            # even if no further improvement happens after resuming.
            raw_ckpt = resume_from.get("best_checkpoint_path")
            if raw_ckpt and Path(raw_ckpt).exists():
                self.best_checkpoint_path = Path(raw_ckpt)
            else:
                # Fallback for checkpoints saved before this field was added:
                # infer from output_dir using known naming conventions.
                for _candidate in ("best_by_auc.pt", "best.pt"):
                    _p = self.output_dir / _candidate
                    if _p.exists():
                        self.best_checkpoint_path = _p
                        log.info("Restored best_checkpoint_path from output_dir: %s", _p.name)
                        break
            log.info(
                "Resuming from epoch %d (best_val_loss=%.4f at epoch %d)",
                start_epoch, self.best_val_loss, self.best_epoch,
            )

        self._fire("on_train_begin")
        t0 = time.perf_counter()

        epoch_iter = tqdm(range(start_epoch, max_epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_iter:
            train_loss = self._train_epoch(epoch, max_epochs)
            val_metrics = self._val_epoch(epoch, max_epochs)

            if self.scheduler is not None:
                self.scheduler.step()

            val_loss = float(val_metrics["loss"])

            if self.checkpoint_metric == "val_auc_roi":
                # ── ROI-modus: split early stopping fra checkpoint-valg ──────
                val_auc_roi = float(val_metrics.get("auc_roc_roi", float("nan")))
                val_f1_roi_opt = float(val_metrics.get("f1_at_opt_roi", float("nan")))

                improved_loss = val_loss < self.best_val_loss
                improved_auc = (
                    not np.isnan(val_auc_roi) and val_auc_roi > self.best_auc_roi
                )
                improved_f1_opt = (
                    not np.isnan(val_f1_roi_opt) and val_f1_roi_opt > self.best_f1_roi_opt
                )

                if improved_loss:
                    self.best_val_loss = val_loss
                if improved_auc:
                    self.best_auc_roi = val_auc_roi
                    self.best_epoch = epoch
                if improved_f1_opt:
                    self.best_f1_roi_opt = val_f1_roi_opt

                row: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": float(val_metrics.get("accuracy", 0.0)),
                    "val_precision": float(val_metrics.get("precision", 0.0)),
                    "val_recall": float(val_metrics.get("recall", 0.0)),
                    "val_f1": float(val_metrics.get("f1", 0.0)),
                    "val_auc_roi": val_auc_roi,
                    "val_avg_precision_roi": float(val_metrics.get("avg_precision_roi", float("nan"))),
                    "val_f1_at_0.5_roi": float(val_metrics.get("f1_at_0.5_roi", float("nan"))),
                    "val_f1_at_opt_roi": val_f1_roi_opt,
                    "val_threshold_opt_roi": float(val_metrics.get("threshold_opt_roi", float("nan"))),
                    "tp": val_metrics.get("tp", 0),
                    "tn": val_metrics.get("tn", 0),
                    "fp": val_metrics.get("fp", 0),
                    "fn": val_metrics.get("fn", 0),
                    "improved": improved_auc,
                    "improved_auc": improved_auc,
                    "improved_f1_opt": improved_f1_opt,
                    "improved_loss": improved_loss,
                    "val_metrics": val_metrics,
                }
                self.history.append(row)

                auc_str = f"{val_auc_roi:.4f}" if not np.isnan(val_auc_roi) else "N/A"
                f1_str  = f"{val_f1_roi_opt:.4f}" if not np.isnan(val_f1_roi_opt) else "N/A"
                epoch_iter.set_postfix(
                    val_loss=f"{val_loss:.4f}",
                    auc_roi=auc_str,
                    best_ep=self.best_epoch,
                )
                log.info(
                    "[epoch %03d] train_loss=%.4f  val_loss=%.4f  auc_roi=%s  f1_opt_roi=%s%s",
                    epoch, train_loss, val_loss, auc_str, f1_str,
                    "  *auc" if improved_auc else "",
                )

                self._fire("on_epoch_end", epoch=epoch, logs=row)

                # Early stopping på AUC-ROI (mode=max); faller tilbake til val_loss hvis AUC er NaN
                es_value = val_auc_roi if not np.isnan(val_auc_roi) else -val_loss
                if self.early_stopping(es_value):
                    log.info("Early stopping at epoch %d (patience=%d, metric=auc_roi)", epoch, self.early_stopping.patience)
                    break

            else:
                # ── Standard modus: val_loss styrer alt (gammel oppførsel) ───
                improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": float(val_metrics.get("accuracy", 0.0)),
                    "val_precision": float(val_metrics.get("precision", 0.0)),
                    "val_recall": float(val_metrics.get("recall", 0.0)),
                    "val_f1": float(val_metrics.get("f1", 0.0)),
                    "tp": val_metrics.get("tp", 0),
                    "tn": val_metrics.get("tn", 0),
                    "fp": val_metrics.get("fp", 0),
                    "fn": val_metrics.get("fn", 0),
                    "improved": improved,
                    "val_metrics": val_metrics,
                }
                self.history.append(row)

                epoch_iter.set_postfix(
                    val_loss=f"{val_loss:.4f}",
                    val_f1=f"{row['val_f1']:.4f}",
                    best=self.best_epoch,
                )
                log.info(
                    "[epoch %03d] train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f%s",
                    epoch, train_loss, val_loss, row["val_acc"], row["val_f1"],
                    "  *" if improved else "",
                )

                self._fire("on_epoch_end", epoch=epoch, logs=row)

                if self.early_stopping(val_loss):
                    log.info("Early stopping at epoch %d (patience=%d)", epoch, self.early_stopping.patience)
                    break

        duration_sec = time.perf_counter() - t0
        final_logs: dict[str, Any] = {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_auc_roi": self.best_auc_roi,
            "best_f1_roi_opt": self.best_f1_roi_opt,
            "best_checkpoint_path": self.best_checkpoint_path,
            "history": self.history,
            "duration_sec": duration_sec,
            "num_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        log.info(
            "Training done: best_epoch=%d  best_val_loss=%.4f  duration=%.1fs",
            self.best_epoch, self.best_val_loss, duration_sec,
        )
        self._fire("on_train_end", logs=final_logs)
        return final_logs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, max_epochs: int) -> float:
        self.model.train()
        running_loss = 0.0
        total = 0

        desc = f"Epoch {epoch}/{max_epochs} train"
        it = tqdm(self.train_loader, desc=desc, unit="batch", leave=False)

        for x, y in it:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(x)
                loss = self.loss_fn(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_n = int(y.size(0))
            running_loss += float(loss.detach().item()) * batch_n
            total += batch_n
            it.set_postfix(loss=f"{running_loss / total:.4f}")

        return running_loss / max(1, total)

    @torch.no_grad()
    def _val_epoch(self, epoch: int, max_epochs: int) -> dict[str, Any]:
        self.model.eval()
        running_loss = 0.0
        total = 0
        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_cube_idxs: list[torch.Tensor] = []

        desc = f"Epoch {epoch}/{max_epochs} val"
        it = tqdm(self.val_loader, desc=desc, unit="batch", leave=False)

        for batch in it:
            if self.roi_val and len(batch) == 3:
                x, y, cube_idx = batch
                all_cube_idxs.append(cube_idx.cpu())
            else:
                x, y = batch

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            batch_n = int(y.size(0))
            running_loss += float(loss.detach().item()) * batch_n
            total += batch_n
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

        if total == 0:
            return {"loss": float("inf"), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        preds = torch.argmax(logits_cat, dim=1).numpy()
        probs = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
        metrics = compute_binary_metrics(targets_cat.numpy(), preds, y_prob=probs)
        metrics["loss"] = running_loss / total

        if self.roi_val and all_cube_idxs:
            cube_idxs_cat = torch.cat(all_cube_idxs, dim=0).numpy()
            roi_m = compute_roi_metrics(cube_idxs_cat, probs, targets_cat.numpy())
            metrics.update(roi_m)

        return metrics

    def _fire(self, hook: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if callable(method):
                try:
                    method(self, **kwargs)
                except Exception as exc:
                    log.error("Callback %s.%s raised: %s", cb.__class__.__name__, hook, exc, exc_info=True)
