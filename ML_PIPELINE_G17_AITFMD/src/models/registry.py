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

from src.models.cnn3d.ad_hybrid_sn import ADHybridSN3DCNN
from src.models.cnn3d.baseline import Baseline3DCNN
from src.models.cnn3d.deeper import Deeper3DCNN
from src.models.cnn3d.multikernel import MultiKernelCNN3D
from src.models.cnn3d.msd_dense import MSDDenseCNN3D
from src.models.cnn3d.resnet_style import ResNet3DCNN
from src.models.cnn3d.se import SECNN3D
from src.models.cnn3d.skip import SkipCNN3D

_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline_3dcnn": Baseline3DCNN,
    "resnet_3dcnn": ResNet3DCNN,
    "deeper_3dcnn": Deeper3DCNN,
    "skip_cnn3d": SkipCNN3D,
    "multikernel_cnn3d": MultiKernelCNN3D,
    "se_cnn3d": SECNN3D,
    "ad_hybrid_sn_3dcnn": ADHybridSN3DCNN,
    "msd_dense_3dcnn": MSDDenseCNN3D,
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
    # configs/models/*.yaml har ofte `architecture:` på rot (ved siden av `model:`), ikke under `model`.
    arch = model_cfg.get("architecture") or cfg.get("architecture") or model_cfg

    in_ch = int(arch.get("in_channels", 1))
    num_classes = int(arch.get("num_classes", 2))
    dropout = float(arch.get("dropout", 0.3))
    _ks = arch.get("kernel_size", 3)
    if isinstance(_ks, (list, tuple)):
        kernel_size = tuple(int(x) for x in _ks)
    else:
        kernel_size = int(_ks)

    cls = _REGISTRY[name]

    if name == "baseline_3dcnn":
        mpl = arch.get("max_pool_layers")
        mpl_kw = {} if mpl is None else {"max_pool_layers": int(mpl)}
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [32, 64, 128])),
            kernel_size=kernel_size,
            dropout=dropout,
            **mpl_kw,
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
        stage_channels = arch.get("stage_channels")
        if stage_channels is None:
            base_channels = int(arch.get("base_channels", 16))
            stage_channels = [base_channels, base_channels * 2, base_channels * 4]
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            stage_channels=list(stage_channels),
            num_blocks=list(arch.get("num_blocks", [2, 2, 2])),
            kernel_size=kernel_size,
            dropout=dropout,
        )

    if name == "skip_cnn3d":
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [16, 32, 64])),
            kernel_size=kernel_size,
            max_pool_layers=int(arch.get("max_pool_layers", 1)),
            dropout=dropout,
        )

    if name == "multikernel_cnn3d":
        def _to_k3(v: object, default: list[int]) -> tuple[int, int, int]:
            if isinstance(v, (list, tuple)):
                return tuple(int(x) for x in v)  # type: ignore[return-value]
            return tuple(default)  # type: ignore[return-value]

        ka = _to_k3(arch.get("kernel_a"), [3, 3, 3])
        kb = _to_k3(arch.get("kernel_b"), [1, 3, 3])
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [16, 32, 64])),
            kernel_size=kernel_size,
            kernel_a=ka,
            kernel_b=kb,
            max_pool_layers=int(arch.get("max_pool_layers", 1)),
            dropout=dropout,
        )

    if name == "se_cnn3d":
        mpl = arch.get("max_pool_layers")
        mpl_kw: dict = {} if mpl is None else {"max_pool_layers": int(mpl)}
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            channels=list(arch.get("channels", [16, 32])),
            kernel_size=kernel_size,
            dropout=dropout,
            se_reduction=int(arch.get("se_reduction", 4)),
            **mpl_kw,
        )

    if name == "msd_dense_3dcnn":
        def _to_k3(v: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
            if isinstance(v, (list, tuple)):
                return tuple(int(x) for x in v)  # type: ignore[return-value]
            return default

        kb_raw = arch.get("kernel_branches")
        if kb_raw is None:
            kernel_branches = ((1, 3, 3), (3, 3, 3), (3, 5, 5))
        else:
            kernel_branches = tuple(_to_k3(row, (3, 3, 3)) for row in kb_raw)  # type: ignore[arg-type]
        pk_raw = arch.get("post_kernel", (3, 3, 3))
        post_kernel = _to_k3(pk_raw, (3, 3, 3))
        sk = arch.get("stem_kernel_size", 3)
        if isinstance(sk, (list, tuple)):
            stem_kernel_size = tuple(int(x) for x in sk)  # type: ignore[assignment]
        else:
            stem_kernel_size = int(sk)
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            stem_channels=int(arch.get("stem_channels", 16)),
            stem_kernel_size=stem_kernel_size,
            kernel_branches=kernel_branches,
            num_layers_per_branch=int(arch.get("num_layers_per_branch", 2)),
            growth_rate=int(arch.get("growth_rate", 8)),
            bottleneck_factor=int(arch.get("bottleneck_factor", 4)),
            post_fusion_channels=int(arch.get("post_fusion_channels", 96)),
            post_kernel=post_kernel,
            dropout=dropout,
        )

    if name == "ad_hybrid_sn_3dcnn":
        ch3 = arch.get("channels_3d", [32, 64])
        ch2 = arch.get("channels_2d", [128, 128])
        k3 = int(arch.get("kernel_size_3d", kernel_size if isinstance(kernel_size, int) else 3))
        k2 = int(arch.get("kernel_size_2d", 3))
        return cls(
            in_channels=in_ch,
            num_classes=num_classes,
            stem_channels=int(arch.get("stem_channels", 16)),
            channels_3d=tuple(int(x) for x in ch3),
            kernel_size_3d=k3,
            transition_3d_out=int(arch.get("transition_3d_out", 32)),
            spectral_bands=int(arch.get("spectral_bands", 16)),
            channels_2d=tuple(int(x) for x in ch2),
            kernel_size_2d=k2,
            spatial_attn_kernel=int(arch.get("spatial_attn_kernel", 7)),
            se_reduction=int(arch.get("se_reduction", 4)),
            pool_after_stem=bool(arch.get("pool_after_stem", True)),
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
