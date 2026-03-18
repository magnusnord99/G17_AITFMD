"""
PyTorch backend for 3D CNN inference.

Laster trent checkpoint og kjører inferanse på patches.
Returnerer sannsynlighet for klasse 1 (anomaly) per patch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import build_model, build_model_from_config


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patches_to_tensor(patches: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Konverter patches (N, H, W, C) til tensor (N, 1, C, H, W) for Conv3d.
    """
    x = torch.from_numpy(patches.astype(np.float32, copy=False))
    # (N, H, W, C) -> (N, 1, C, H, W)
    x = x.permute(0, 3, 1, 2).unsqueeze(1)  # (N, C, H, W) -> (N, 1, C, H, W)
    return x.to(device, non_blocking=True)


def predict_pytorch(
    patches: np.ndarray,
    checkpoint_path: str | Path,
    model_config_path: str | Path | None = None,
    project_root: Path | None = None,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Kjør 3D CNN på patches og returner anomaly-sannsynlighet per patch.

    Args:
        patches: (N, H, W, C) float32, typisk (N, 64, 64, 16)
        checkpoint_path: Sti til .pt checkpoint
        model_config_path: Sti til model YAML (hvis None, leses fra checkpoint)
        project_root: Prosjektrot for relative stier
        device: torch device (hvis None, velges automatisk)
        batch_size: Batch-størrelse for inferanse

    Returns:
        scores: (N,) float32 i [0, 1] – sannsynlighet for anomaly per patch
    """
    if len(patches) == 0:
        return np.array([], dtype=np.float32)

    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent.parent
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (root / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = device or _pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Modellconfig: fra checkpoint eller eksplisitt path
    model_cfg_path = model_config_path
    if model_cfg_path is None and "model_config_path" in ckpt:
        model_cfg_path = Path(ckpt["model_config_path"])
    if model_cfg_path is None:
        raise ValueError("model_config_path required (in checkpoint or as argument)")

    model_cfg_path = Path(model_cfg_path)
    if not model_cfg_path.is_absolute():
        model_cfg_path = (root / model_cfg_path).resolve()

    model = build_model_from_config(model_cfg_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    scores_list: list[float] = []
    n = len(patches)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = patches[start:end]
            x = _patches_to_tensor(batch, device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            # Klasse 1 = anomaly
            anomaly_prob = probs[:, 1].cpu().numpy()
            scores_list.extend(anomaly_prob.tolist())

    return np.array(scores_list, dtype=np.float32)
