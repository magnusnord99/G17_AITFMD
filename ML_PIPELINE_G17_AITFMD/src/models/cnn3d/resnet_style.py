"""ResNet-inspirert 3D CNN med residual blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conv_utils import make_groupnorm


def _conv3x3(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Conv3d:
    return nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)


class ResBlock3D(nn.Module):
    """Residual block: conv-norm-relu-dropout-conv-norm + skip."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, block_dropout: float = 0.0):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, kernel_size)
        self.bn1 = make_groupnorm(out_channels)
        self.drop = nn.Dropout(p=block_dropout) if block_dropout > 0.0 else nn.Identity()
        self.conv2 = _conv3x3(out_channels, out_channels, kernel_size)
        self.bn2 = make_groupnorm(out_channels)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                make_groupnorm(out_channels),
            )
        else:
            self.project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(x)
        x = self.drop(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        return nn.functional.relu(x + residual)


class ResNet3DCNN(nn.Module):
    """3D CNN med residual blocks per stage."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        stage_channels: list[int] = (16, 32, 64),
        num_blocks: list[int] = (2, 2, 2),
        kernel_size: int = 3,
        dropout: float = 0.3,
        block_dropout: float = 0.0,
    ):
        super().__init__()
        if len(stage_channels) != len(num_blocks):
            raise ValueError(
                "stage_channels and num_blocks must have the same length "
                f"(got {len(stage_channels)} and {len(num_blocks)})"
            )

        first_channels = stage_channels[0]
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, first_channels, kernel_size, padding=kernel_size // 2),
            make_groupnorm(first_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        stages = []
        prev_channels = first_channels
        for out_channels, n_stage_blocks in zip(stage_channels, num_blocks):
            stages.append(ResBlock3D(prev_channels, out_channels, kernel_size, block_dropout=block_dropout))
            for _ in range(n_stage_blocks - 1):
                stages.append(ResBlock3D(out_channels, out_channels, kernel_size, block_dropout=block_dropout))
            stages.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            prev_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.stages(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
