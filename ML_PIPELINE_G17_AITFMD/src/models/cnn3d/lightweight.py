"""Lightweight 3D CNN – færre kanaler, raskere."""

from __future__ import annotations

from src.models.conv_utils import Kernel3D
from src.models.cnn3d.baseline import Baseline3DCNN


class Lightweight3DCNN(Baseline3DCNN):
    """Samme arkitektur som baseline, men med færre kanaler."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: list[int] = (16, 32, 64),
        kernel_size: Kernel3D = 3,
        dropout: float = 0.25,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
