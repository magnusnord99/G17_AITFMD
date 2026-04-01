"""Skip-connection 3D CNN (baseline-kompatibel residual variant).

Arkitektur:
  Entry blokk  : valid Conv3d (reduserer D/H/W) → GroupNorm → ReLU → MaxPool(1,2,2)
  Residual stages: N × SkipBlock3D (same-padding, bevarer romlig dim)
                   + valgfri MaxPool(1,2,2) mellom stages
  Hode         : AdaptiveAvgPool3d(1) → Dropout → Linear

Forskjell fra baseline_3dcnn:
  - Same-padding 3D-convs inni residualblokker i stedet for valid convs
  - Skip-connection (+ shortcut-projeksjoner om kanaler endres)
  - Dypere kanalstack: channels = [entry, stage1, stage2, ...]

Forskjell fra resnet_3dcnn:
  - GroupNorm (ikke BatchNorm3d) — fungerer bedre med liten batch
  - Spatial-only pooling (1,2,2), ikke isotropisk MaxPool(2)
  - Valid conv i entry (matcher baseline geometri)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_utils import Kernel3D, make_groupnorm, normalize_conv3d_kernel_size


class SkipBlock3D(nn.Module):
    """Residual blokk med same-padding convs og GroupNorm.

    Conv3d(3,same) → GN → ReLU → Conv3d(3,same) → GN
    + skip (1×1×1 projeksjonsconv om in_ch ≠ out_ch)
    → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = make_groupnorm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = make_groupnorm(out_ch)
        # Shortcut: projeksjonsconv om kanalantall endres
        self.proj: nn.Module = (
            nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), make_groupnorm(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + res, inplace=True)


class SkipCNN3D(nn.Module):
    """3D CNN med residual skip-connections.

    Parametere:
        in_channels     : spektrale inngangsbånd (D), vanligvis 1 (uflattened cube)
        num_classes     : antall klassifiseringsklasser
        channels        : [entry_ch, stage1_ch, stage2_ch, ...]
                          channels[0] brukes av entry-conv, resten av residualblokker
        kernel_size     : kernel for entry valid conv, f.eks. (4,3,3)
        max_pool_layers : antall MaxPool(1,2,2) som legges inn etter residualblokker
                          (1 = pool etter blokk 1, osv.)
        dropout         : dropout-rate før FC-lag
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: Kernel3D = (3, 3, 3),
        max_pool_layers: int = 1,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels må ha minst 2 elementer: [entry_ch, stage1_ch, ...]")

        ks = normalize_conv3d_kernel_size(kernel_size)

        # Entry: valid conv + pool (komprimerer D og H/W på samme måte som baseline)
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=ks, padding=0, bias=False),
            make_groupnorm(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Residual stages (same-padding, bevarer romlig dimensjon gjennom blokken)
        stages: list[nn.Module] = []
        for i, out_ch in enumerate(channels[1:], start=1):
            stages.append(SkipBlock3D(channels[i - 1], out_ch))
            if i < max_pool_layers:
                stages.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.stages(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
