"""
Modell-registry: instantier CNN fra config ved navn.

Bruk:
    from src.models import build_model, build_model_from_config
    model = build_model("baseline_3dcnn", config_dict)
    model = build_model_from_config("configs/models/baseline_3dcnn.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from src.models.cnn3d.baseline import Baseline3DCNN
from src.models.cnn3d.deeper import Deeper3DCNN
from src.models.cnn3d.lightweight import Lightweight3DCNN
from src.models.cnn3d.resnet_style import ResNet3DCNN

_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline_3dcnn": Baseline3DCNN,
    "lightweight_3dcnn": Lightweight3DCNN,
    "resnet_3dcnn": ResNet3DCNN,
    "deeper_3dcnn": Deeper3DCNN,
}


def build_model(name: str, cfg: dict[str, Any]) -> nn.Module:
    """
    Bygg modell fra navn og config.

    Args:
        name: Modellnavn (f.eks. "baseline_3dcnn")
        cfg: Hele config-dict eller model/architecture-delen

    Returns:
        nn.Module klar for trening
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY)}")

    model_cfg = cfg.get("model", cfg)
    arch = model_cfg.get("architecture", model_cfg)

    in_ch = int(arch.get("in_channels", 1))
    num_classes = int(arch.get("num_classes", 2))
    dropout = float(arch.get("dropout", 0.3))
    kernel_size = int(arch.get("kernel_size", 3))

    cls = _REGISTRY[name]

    if name == "baseline_3dcnn":
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [32, 64, 128])),
            kernel_size=kernel_size,
            dropout=dropout,
        )
    if name == "lightweight_3dcnn":
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [16, 32, 64])),
            kernel_size=kernel_size,
            dropout=dropout,
        )
    if name == "deeper_3dcnn":
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [32, 64, 128, 256, 256])),
            kernel_size=kernel_size,
            dropout=dropout,
        )
    if name == "resnet_3dcnn":
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            base_channels=int(arch.get("base_channels", 32)),
            num_blocks=list(arch.get("num_blocks", [2, 2, 2])),
            kernel_size=kernel_size,
            dropout=dropout,
        )

    raise ValueError(f"No builder for model: {name}")


def list_models() -> list[str]:
    """Returner alle registrerte modellnavn."""
    return list(_REGISTRY.keys())


def build_model_from_config(config_path: str | Path) -> nn.Module:
    """
    Last config fra fil og bygg modell.

    Args:
        config_path: Sti til YAML (f.eks. configs/models/baseline_3dcnn.yaml)

    Returns:
        nn.Module
    """
    path = Path(config_path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parent.parent.parent
        path = root / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = cfg.get("model", {}).get("name") or path.stem
    return build_model(name, cfg)
