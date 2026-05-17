"""torchinfo-sammendrag for alle fire 3D-CNN-modeller.

Kjøres fra prosjektrot:
    python docs/visualizations/model_summary.py

Input: (B=1, C=1, D=16, H=64, W=64) — matcher wavelet16 + patch 64×64.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchinfo import summary

from src.models.cnn3d.baseline import Baseline3DCNN
from src.models.cnn3d.resnet_style import ResNet3DCNN
from src.models.cnn3d.msd_dense import MSDDenseCNN3D
from src.models.cnn3d.ad_hybrid_sn import ADHybridSN3DCNN

INPUT = (1, 1, 16, 64, 64)  # (B, C, D, H, W)

MODELS = [
    (
        "Baseline3DCNN",
        Baseline3DCNN(
            in_channels=1, num_classes=2,
            channels=(16, 32), kernel_size=[4, 3, 3], dropout=0.42, max_pool_layers=2,
        ),
    ),
    (
        "ResNet3DCNN",
        ResNet3DCNN(
            in_channels=1, num_classes=2,
            stage_channels=[16, 32, 64], num_blocks=[2, 2, 2],
            kernel_size=3, dropout=0.45, block_dropout=0.10,
        ),
    ),
    (
        "MSDDenseCNN3D",
        MSDDenseCNN3D(
            in_channels=1, num_classes=2,
            stem_channels=16, stem_kernel_size=3,
            kernel_branches=((1, 3, 3), (3, 3, 3), (3, 5, 5)),
            num_layers_per_branch=2, growth_rate=4, bottleneck_factor=4,
            post_fusion_channels=48, post_kernel=(3, 3, 3), dropout=0.30,
        ),
    ),
    (
        "ADHybridSN3DCNN",
        ADHybridSN3DCNN(
            in_channels=1, num_classes=2,
            stem_channels=16, channels_3d=(32, 64), kernel_size_3d=3,
            transition_3d_out=32, spectral_bands=16,
            channels_2d=(128, 128), kernel_size_2d=3,
            spatial_attn_kernel=7, se_reduction=4,
            pool_after_stem=True, dropout=0.30,
        ),
    ),
]

SEP = "=" * 80

for name, model in MODELS:
    print(f"\n{SEP}")
    print(f"  {name}")
    print(SEP)
    summary(
        model,
        input_size=INPUT,
        col_names=["input_size", "output_size", "num_params"],
        depth=4,
        verbose=1,
    )
