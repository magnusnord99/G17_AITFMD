"""MSD 3D-CNN: tre parallelle dense-blokker med ulike 3D-kjernestørrelser.

Inspirert av multi-scale dense branches for HSI-klassifisering: hver gren kjører
DenseNet-BC-lignende lag (1×1×1 bottleneck + k×k×k) med fast kernel per gren,
fusjon ved kanal-konkatenering, deretter to konvolusjonslag, global pooling og FC.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_utils import Kernel3D, make_groupnorm, normalize_conv3d_kernel_size, same_pad


def _stem_kernel(kernel_size: Kernel3D) -> tuple[int, int, int]:
    return normalize_conv3d_kernel_size(kernel_size)


class _DenseLayer3D(nn.Module):
    """Ett lag i en dense-blokk: BN–ReLU–Conv 1³ – BN–ReLU–Conv k³ → growth_rate kanaler."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        kernel_size: tuple[int, int, int],
        bottleneck_factor: int = 4,
    ) -> None:
        super().__init__()
        bn_size = bottleneck_factor * growth_rate
        self.bn1 = make_groupnorm(num_input_features)
        self.conv1 = nn.Conv3d(num_input_features, bn_size, kernel_size=1, bias=False)
        self.bn2 = make_groupnorm(bn_size)
        pad = same_pad(kernel_size)
        self.conv2 = nn.Conv3d(bn_size, growth_rate, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)


class DenseBlock3DBranch(nn.Module):
    """Dense-blokk med `num_layers` lag og gitt 3D-kjerne for alle lag i blokken."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        kernel_size: tuple[int, int, int],
        bottleneck_factor: int = 4,
    ) -> None:
        super().__init__()
        ks = tuple(int(k) for k in kernel_size)
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(num_layers):
            layers.append(
                _DenseLayer3D(
                    c,
                    growth_rate,
                    ks,
                    bottleneck_factor=bottleneck_factor,
                )
            )
            c += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MSDDenseCNN3D(nn.Module):
    """Delt stem → tre parallelle dense-blokker → concat → to conv → GAP → FC."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        stem_channels: int = 16,
        stem_kernel_size: Kernel3D = 3,
        kernel_branches: tuple[tuple[int, int, int], ...] = (
            (1, 3, 3),
            (3, 3, 3),
            (3, 5, 5),
        ),
        num_layers_per_branch: int = 2,
        growth_rate: int = 8,
        bottleneck_factor: int = 4,
        post_fusion_channels: int = 96,
        post_kernel: tuple[int, int, int] = (3, 3, 3),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(kernel_branches) != 3:
            raise ValueError(f"MSD krever nøyaktig 3 kernel_branches, fikk {len(kernel_branches)}")
        sk = _stem_kernel(stem_kernel_size)
        pad = same_pad(sk)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=sk, padding=pad, bias=False),
            make_groupnorm(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        kerns = [tuple(int(x) for x in k) for k in kernel_branches]
        self.branches = nn.ModuleList(
            [
                DenseBlock3DBranch(
                    stem_channels,
                    num_layers_per_branch,
                    growth_rate,
                    kerns[i],
                    bottleneck_factor=bottleneck_factor,
                )
                for i in range(3)
            ]
        )
        concat_ch = sum(b.out_channels for b in self.branches)
        pk = tuple(int(x) for x in post_kernel)
        pp = same_pad(pk)
        self.post_fusion = nn.Sequential(
            nn.Conv3d(concat_ch, post_fusion_channels, kernel_size=1, bias=False),
            make_groupnorm(post_fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(post_fusion_channels, post_fusion_channels, kernel_size=pk, padding=pp, bias=False),
            make_groupnorm(post_fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(post_fusion_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        outs = [self.branches[i](x) for i in range(3)]
        dhw = [tuple(t.shape[2:]) for t in outs]
        if not (dhw[0] == dhw[1] == dhw[2]):
            raise RuntimeError(
                "MSD-grener har ulik romlig/spektral størrelse etter dense-blokker; "
                f"shapes={dhw}. Juster kernel_branches eller bruk eksplisitt stem."
            )
        x = torch.cat(outs, dim=1)
        x = self.post_fusion(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
