"""ResNet-inspirert 3D CNN med residual blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, 3, padding=1)


class ResBlock3D(nn.Module):
    """Residual block: conv-bn-relu-conv-bn + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.functional.relu(x + residual)


class ResNet3DCNN(nn.Module):
    """3D CNN med residual blocks per stage."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_blocks: list[int] = (2, 2, 2),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        stages = []
        ch = base_channels
        for n_blocks in num_blocks:
            for _ in range(n_blocks):
                stages.append(ResBlock3D(ch))
            stages.append(nn.MaxPool3d(2))
            # Double channels after each stage (optional; keep same for simplicity)
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.stages(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
