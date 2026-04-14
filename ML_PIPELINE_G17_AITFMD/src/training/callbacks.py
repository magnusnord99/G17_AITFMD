"""Training callbacks — hooks fired by the Trainer at key lifecycle events."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.training.trainer import Trainer

log = get_logger(__name__)


class BaseCallback:
    """No-op base class.  Subclass and override only the hooks you need."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_train_end(self, trainer: "Trainer", logs: dict[str, Any]) -> None:
        pass


class CheckpointCallback(BaseCallback):
    """Save best and last checkpoints after each epoch.

    The checkpoint dict schema is::

        {
            "epoch": int,
            "model_state_dict": ...,
            "optimizer_state_dict": ...,
            "model_config_path": str,   # embedded so Evaluator can reload
            "train_config_path": str,
            "val_metrics": dict,
        }

    Args:
        checkpoint_dir: Directory where ``.pt`` files are saved.
        model_name: Base name used for filenames (e.g. ``"baseline_3dcnn"``).
        run_id: Unique run identifier (e.g. ``"20250410_120000"``).
        model_config_path: Path to the model YAML (embedded in checkpoint).
        train_config_path: Path to the train YAML (embedded in checkpoint).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str,
        run_id: str,
        model_config_path: Path,
        train_config_path: Path,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.run_id = run_id
        self.model_config_path = model_config_path
        self.train_config_path = train_config_path

        self.best_path = self.checkpoint_dir / "best.pt"
        self.best_by_auc_path = self.checkpoint_dir / "best_by_auc.pt"
        self.best_by_f1_opt_path = self.checkpoint_dir / "best_by_f1_opt.pt"
        self.best_by_loss_path = self.checkpoint_dir / "best_by_loss.pt"
        self.last_path = self.checkpoint_dir / "last.pt"

    def _make_state(self, trainer: "Trainer", epoch: int, logs: dict[str, Any]) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            "model_config_path": str(self.model_config_path),
            "train_config_path": str(self.train_config_path),
            "val_metrics": logs.get("val_metrics", {}),
            "best_val_loss": trainer.best_val_loss,
            "best_epoch": trainer.best_epoch,
            "best_auc_roi": trainer.best_auc_roi,
            "best_f1_roi_opt": trainer.best_f1_roi_opt,
            "history": trainer.history,
        }

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict[str, Any]) -> None:
        state = self._make_state(trainer, epoch, logs)
        torch.save(state, self.last_path)

        if trainer.checkpoint_metric == "val_auc_roi":
            # ── ROI-modus: tre separate checkpoint-filer ──────────────────
            if logs.get("improved_auc", False):
                torch.save(state, self.best_by_auc_path)
                trainer.best_checkpoint_path = self.best_by_auc_path
                log.info("Checkpoint saved (best_by_auc): %s", self.best_by_auc_path.name)
            if logs.get("improved_f1_opt", False):
                torch.save(state, self.best_by_f1_opt_path)
                log.info("Checkpoint saved (best_by_f1_opt): %s", self.best_by_f1_opt_path.name)
            if logs.get("improved_loss", False):
                torch.save(state, self.best_by_loss_path)
                log.info("Checkpoint saved (best_by_loss): %s", self.best_by_loss_path.name)
        else:
            # ── Standard modus: gammel oppførsel, én best.pt ──────────────
            if logs.get("improved", False):
                torch.save(state, self.best_path)
                trainer.best_checkpoint_path = self.best_path
                log.info("Checkpoint saved (best): %s", self.best_path.name)

    def on_train_end(self, trainer: "Trainer", logs: dict[str, Any]) -> None:
        log.info("Best checkpoint: %s", trainer.best_checkpoint_path)
        log.info("Last checkpoint: %s", self.last_path)


class EvalCallback(BaseCallback):
    """Run full ROI-level evaluation on one or more splits after training ends.

    This is the auto-eval hook.  It fires exactly once in ``on_train_end`` so
    that expensive inference only runs after the best checkpoint is written.

    Args:
        cfg: Full parsed ``train.yaml`` dict (passed through to ``Evaluator``).
        output_dir: Base output directory (split sub-dirs are created inside).
        splits: Splits to evaluate, e.g. ``["val", "test"]``.
        batch_size: Override batch size for inference.
        patient_level: Pass through to ``Evaluator``.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: Path,
        splits: list[str] | None = None,
        batch_size: int | None = None,
        patient_level: bool = False,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.splits = splits or ["val", "test"]
        self.batch_size = batch_size
        self.patient_level = patient_level

    def on_train_end(self, trainer: "Trainer", logs: dict[str, Any]) -> None:
        if trainer.best_checkpoint_path is None:
            log.warning("EvalCallback: no best checkpoint found, skipping eval.")
            return

        from src.evaluation.evaluator import Evaluator  # local import avoids circular dep

        # Embed run_id into cfg so Evaluator can include it in reports
        cfg_with_run_id = {**self.cfg, "_run_id": trainer.run_id}

        evaluator = Evaluator(
            checkpoint_path=trainer.best_checkpoint_path,
            cfg=cfg_with_run_id,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            patient_level=self.patient_level,
        )

        for split in self.splits:
            log.info("EvalCallback: running eval for split=%s", split)
            try:
                metrics = evaluator.run(split)
                auc = metrics.get("auc_roc", float("nan"))
                f1 = metrics.get("metrics_at_threshold_0.5", {}).get("f1", float("nan"))
                log.info(
                    "Auto-eval %s: auc_roc=%.4f  f1@0.5=%.4f",
                    split, auc, f1,
                )
            except Exception as exc:
                log.error("EvalCallback: eval failed for split=%s: %s", split, exc, exc_info=True)
