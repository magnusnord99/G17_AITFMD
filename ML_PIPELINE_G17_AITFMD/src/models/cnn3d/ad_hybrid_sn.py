"""AD-HybridSN-inspirert variant: to 3D-blokker med kanal-attention, overgang til 2D, to 2D-blokker med romlig attention.

Bruk GroupNorm, residualforbindelser og konfigurerbar dropout. Input: (B, 1, D, H, W) som øvrige 3D-CNN i prosjektet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_utils import make_groupnorm, normalize_conv3d_kernel_size, same_pad


class ChannelAttention3D(nn.Module):
    """SE-lignende kanal-attention på (B, C, D, H, W)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.gap(x).flatten(1)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class SpatialAttention2D(nn.Module):
    """Romlig attention (CBAM-lignende): avg/max over kanal → concat → conv → sigmoid."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        z = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(z))
        return x * att


class DepthwiseSeparable2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ThreeDResidualBlock(nn.Module):
    """To Conv3d + residual + ReLU, deretter kanal-attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        ks = normalize_conv3d_kernel_size(kernel_size)
        pad = same_pad(ks)
        self.conv1 = nn.Conv3d(in_ch, out_ch, ks, padding=pad, bias=False)
        self.gn1 = make_groupnorm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, ks, padding=pad, bias=False)
        self.gn2 = make_groupnorm(out_ch)
        self.proj = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else None
        )
        self.ca = ChannelAttention3D(out_ch, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.proj is not None else x
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return self.ca(out)


class TwoDResidualDSBlock(nn.Module):
    """Depthwise-separable conv + GN + ReLU + residual + spatial attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        sa_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.ds = DepthwiseSeparable2d(in_ch, out_ch, kernel_size=kernel_size)
        self.gn = make_groupnorm(out_ch)
        self.proj = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else None
        )
        self.sa = SpatialAttention2D(kernel_size=sa_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.proj is not None else x
        out = self.ds(x)
        out = self.gn(out)
        out = F.relu(out + identity, inplace=True)
        return self.sa(out)


class ADHybridSN3DCNN(nn.Module):
    """To 3D-blokker (kanal-attention) → 3D→2D → to 2D-blokker (romlig attention) → GAP → FC."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        stem_channels: int = 16,
        channels_3d: tuple[int, int] = (32, 64),
        kernel_size_3d: int = 3,
        transition_3d_out: int = 32,
        spectral_bands: int = 16,
        channels_2d: tuple[int, int] = (128, 128),
        kernel_size_2d: int = 3,
        spatial_attn_kernel: int = 7,
        se_reduction: int = 4,
        pool_after_stem: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        ks3 = normalize_conv3d_kernel_size(kernel_size_3d)
        p3 = same_pad(ks3)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, ks3, padding=p3, bias=False),
            make_groupnorm(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.pool_stem = (
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            if pool_after_stem
            else nn.Identity()
        )

        c0, c1 = channels_3d
        self.block3d_1 = ThreeDResidualBlock(
            stem_channels, c0, kernel_size=kernel_size_3d, se_reduction=se_reduction
        )
        self.block3d_2 = ThreeDResidualBlock(
            c0, c1, kernel_size=kernel_size_3d, se_reduction=se_reduction
        )

        self.transition_conv = nn.Sequential(
            nn.Conv3d(c1, transition_3d_out, kernel_size=1, bias=False),
            make_groupnorm(transition_3d_out),
            nn.ReLU(inplace=True),
        )
        self.transition_3d_out = transition_3d_out
        self.spectral_bands = spectral_bands

        flat_in = transition_3d_out * spectral_bands
        c2a, c2b = channels_2d
        self.block2d_1 = TwoDResidualDSBlock(
            flat_in,
            c2a,
            kernel_size=kernel_size_2d,
            sa_kernel=spatial_attn_kernel,
        )
        self.block2d_2 = TwoDResidualDSBlock(
            c2a,
            c2b,
            kernel_size=kernel_size_2d,
            sa_kernel=spatial_attn_kernel,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c2b, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool_stem(x)
        x = self.block3d_1(x)
        x = self.block3d_2(x)
        x = self.transition_conv(x)

        b, c, d, h, w = x.shape
        if d != self.spectral_bands:
            raise RuntimeError(
                f"Spektral dybde {d} matcher ikke forventet spectral_bands={self.spectral_bands}. "
                "Sett architecture.spectral_bands i YAML lik antall bånd etter preprocessing."
            )
        x = x.reshape(b, c * d, h, w)

        x = self.block2d_1(x)
        x = self.block2d_2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
