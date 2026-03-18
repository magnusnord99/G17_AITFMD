"""Baseline 3D CNN – enkel conv-stack med global pooling."""

from __future__ import annotations

import torch
import torch.nn as nn


class Baseline3DCNN(nn.Module):
    """Enkel 3D CNN: conv-blokker → global avg pool → FC."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = in_channels
        for c in channels:
            layers += [
                nn.Conv3d(prev, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm3d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) for torch.nn.Conv3d
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
