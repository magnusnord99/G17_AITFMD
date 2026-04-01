"""Squeeze-and-Excite 3D CNN — kanal-attention variant av baseline.

Grunnarkitekturen er identisk med baseline_3dcnn (valid conv, GroupNorm,
MaxPool(1,2,2)), men med en lett Squeeze-and-Excite (SE) attention-modul
etter ReLU i hvert conv-lag.

SE-modulen (Hu et al., 2018 — «Squeeze-and-Excitation Networks»):
  Squeeze : AdaptiveAvgPool3d(1) → kanalvektor av størrelse C
  Excite  : Linear(C → C//r) → ReLU → Linear(C//r → C) → Sigmoid
  Scale   : gang sigmoid-vekter inn i feature-map (per kanal)

Effekten er at nettverket lærer «hva det skal se på» kanalvis, uten å
legge til mange parametere (typisk < 2 % av totalparametrene).

Parametere:
  se_reduction (r) : komprimeringsfaktor i SE-laget (standard 4).
                     Lavere verdi = flere parametere og mer kapasitet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conv_utils import Kernel3D, make_groupnorm, normalize_conv3d_kernel_size


class SEModule3D(nn.Module):
    """Lett kanal-attention (Squeeze-and-Excite) for 3D feature maps.

    Input/output: (B, C, D, H, W) — ingen endring i form, bare skalering per kanal.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.gap(x).flatten(1)          # (B, C)
        scale = self.fc(scale)                  # (B, C) — sigmoid-vekter
        return x * scale.view(-1, x.shape[1], 1, 1, 1)


class SECNN3D(nn.Module):
    """3D CNN med SE kanal-attention etter hvert conv-lag.

    Geometri per blokk:
      Conv3d(valid) → GroupNorm → ReLU → SEModule3D → [MaxPool3d(1,2,2)]

    Identisk med baseline_3dcnn bortsett fra at SEModule3D skalerer
    kanalene etter ReLU. Dette gir lite ekstra kostnad men bedre
    informasjonsseleksjon.

    Parametere:
        in_channels     : spektrale inngangsbånd (D), vanligvis 1
        num_classes     : antall klasser
        channels        : kanalantall per blokk, f.eks. [16, 32]
        kernel_size     : valid conv-kernel, f.eks. (4,3,3)
        max_pool_layers : antall blokker som avslutter med MaxPool(1,2,2)
                          None = alle blokker
        dropout         : dropout-rate
        se_reduction    : SE-komprimering r (channels → channels//r → channels)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (16, 32),
        kernel_size: Kernel3D = (4, 3, 3),
        max_pool_layers: int | None = None,
        dropout: float = 0.42,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        n_blocks = len(channels)
        if max_pool_layers is None:
            max_pool_layers = n_blocks
        if not (1 <= max_pool_layers <= n_blocks):
            raise ValueError(
                f"max_pool_layers må være i [1, {n_blocks}], fikk {max_pool_layers}"
            )

        ks = normalize_conv3d_kernel_size(kernel_size)
        layers: list[nn.Module] = []
        prev = in_channels
        for i, c in enumerate(channels):
            layers += [
                nn.Conv3d(prev, c, kernel_size=ks, padding=0, bias=False),
                make_groupnorm(c),
                nn.ReLU(inplace=True),
                SEModule3D(c, reduction=se_reduction),
            ]
            if i < max_pool_layers:
                layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            prev = c

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
