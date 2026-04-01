"""Multi-kernel 3D CNN med parallelle conv-grener.

Hver blokk bruker to parallelle konvolusjonsgrener med ulike kernel sizes
og concatenerer resultatet. Modellen kan dermed fange mønster på ulike
romlige og spektrale skalaer samtidig — inspirert av Inception-arkitektur.

Arkitektur:
  Entry blokk         : valid Conv3d → GroupNorm → ReLU → MaxPool(1,2,2)
  MultiKernelBlock ×N : to parallelle same-padding conv-grener → cat → GN → ReLU
                        + valgfri MaxPool(1,2,2) mellom blokker
  Hode                : AdaptiveAvgPool3d(1) → Dropout → Linear

Standardgrener (konfigurerbart via YAML):
  kernel_a = (3,3,3) — kubisk 3D-kontekst
  kernel_b = (1,3,3) — kun romlig (H×W), spektral D ignoreres

Hvert `out_ch` i channels[1:] deles likt mellom de to grenene (out_ch // 2 pr. gren),
så out_ch må være et partall.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv_utils import Kernel3D, make_groupnorm, normalize_conv3d_kernel_size, same_pad


class MultiKernelBlock3D(nn.Module):
    """Parallell conv-blokk med to ulike kernel sizes.

    Branch A: Conv3d(in_ch, out_ch//2, kernel_a, same-padding) → GN → ReLU
    Branch B: Conv3d(in_ch, out_ch//2, kernel_b, same-padding) → GN → ReLU
    Concat → GroupNorm(out_ch) → ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_a: tuple[int, int, int] = (3, 3, 3),
        kernel_b: tuple[int, int, int] = (1, 3, 3),
    ) -> None:
        super().__init__()
        if out_ch % 2 != 0:
            raise ValueError(f"out_ch må være partall for MultiKernelBlock, fikk {out_ch}")
        branch_ch = out_ch // 2
        self.branch_a = nn.Conv3d(
            in_ch, branch_ch, kernel_a, padding=same_pad(kernel_a), bias=False
        )
        self.branch_b = nn.Conv3d(
            in_ch, branch_ch, kernel_b, padding=same_pad(kernel_b), bias=False
        )
        self.gn = make_groupnorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        return F.relu(self.gn(torch.cat([a, b], dim=1)), inplace=True)


class MultiKernelCNN3D(nn.Module):
    """3D CNN med parallelle conv-grener per blokk.

    Parametere:
        in_channels     : spektrale inngangsbånd (D), vanligvis 1
        num_classes     : antall klasser
        channels        : [entry_ch, stage1_ch, stage2_ch, ...]
                          channels[0] til entry-conv, resten til MultiKernelBlock
                          NB: channels[1:] må være partall (deles mellom to grener)
        kernel_size     : kernel for entry valid conv, f.eks. (4,3,3)
        kernel_a        : kernel size for gren A i parallelle blokker
        kernel_b        : kernel size for gren B (f.eks. (1,3,3) = kun spatial)
        max_pool_layers : antall MaxPool(1,2,2) etter blokker i stages
        dropout         : dropout-rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: Kernel3D = (4, 3, 3),
        kernel_a: tuple[int, int, int] = (3, 3, 3),
        kernel_b: tuple[int, int, int] = (1, 3, 3),
        max_pool_layers: int = 1,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels må ha minst 2 elementer: [entry_ch, stage1_ch, ...]")

        ks = normalize_conv3d_kernel_size(kernel_size)
        ka = tuple(int(k) for k in kernel_a)
        kb = tuple(int(k) for k in kernel_b)

        # Entry: valid conv + pool
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=ks, padding=0, bias=False),
            make_groupnorm(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Parallelle blokker
        stages: list[nn.Module] = []
        for i, out_ch in enumerate(channels[1:], start=1):
            stages.append(MultiKernelBlock3D(channels[i - 1], out_ch, ka, kb))  # type: ignore[arg-type]
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
