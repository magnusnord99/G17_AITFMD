"""Simple convolutional autoencoder for HSI patch compression."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Spectral autoencoder for HSI patches.
    Uses 1x1 convolutions only — compresses along the spectral axis per pixel,
    analogous to PCA/wavelets (no spatial mixing).
    """

    def __init__(self, in_channels: int = 275, latent_channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        out = self.decode(z)
        return out
