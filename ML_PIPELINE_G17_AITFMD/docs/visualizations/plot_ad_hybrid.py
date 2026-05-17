"""PlotNeuralNet diagram for ADHybridSN3DCNN.

ADHybridSN3DCNN: Stem → pool → 3D ResBlock+ChannelAttn x2
                 → 1x1 Transition → reshape 3D→2D
                 → 2D DS-ResBlock+SpatialAttn x2 → GAP → FC
Default: stem=16, 3D blocks=(32,64), transition=32, spectral=16,
         flat=512ch, 2D blocks=(128,128)
"""
import sys
sys.path.insert(0, '/Users/magnusnordmo/PlotNeuralNet')
from pycore.tikzeng import *

PLOTNN = '/Users/magnusnordmo/PlotNeuralNet'

arch = [
    to_head(PLOTNN),
    to_cor(),
    to_begin(),

    # Input
    to_Conv("input", s_filer=1, n_filer=1, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=1, caption="Input"),

    # Stem: Conv3D(1→16) + GN + ReLU
    to_Conv("stem", s_filer=16, n_filer=16, offset="(2,0,0)", to="(input-east)", height=40, depth=40, width=2, caption="Stem\\\\Conv-16"),
    to_connection("input", "stem"),
    to_Pool("stempool", offset="(0,0,0)", to="(stem-east)", height=32, depth=32, width=1, caption="Pool"),

    # 3D Block 1: ResBlock(16→32) + ChannelAttention
    to_ConvConvRelu("b3d1", s_filer=32, n_filer=(32,32), offset="(1,0,0)", to="(stempool-east)", height=32, depth=32, width=(3,3), caption="3D Res\\\\+ChAttn\\\\32"),
    to_connection("stempool", "b3d1"),

    # 3D Block 2: ResBlock(32→64) + ChannelAttention
    to_ConvConvRelu("b3d2", s_filer=64, n_filer=(64,64), offset="(1,0,0)", to="(b3d1-east)", height=28, depth=28, width=(4,4), caption="3D Res\\\\+ChAttn\\\\64"),
    to_connection("b3d1", "b3d2"),

    # Transition conv 1x1: (64→32)
    to_Conv("trans", s_filer=32, n_filer=32, offset="(1,0,0)", to="(b3d2-east)", height=24, depth=24, width=2, caption="Trans\\\\1x1-32"),
    to_connection("b3d2", "trans"),

    # Reshape 3D→2D (shown as a pool/squeeze block)
    to_Pool("reshape", offset="(0,0,0)", to="(trans-east)", height=20, depth=20, width=1, caption="3D→2D\\\\512ch"),

    # 2D Block 1: DS-ResBlock(512→128) + SpatialAttention
    to_ConvConvRelu("b2d1", s_filer=128, n_filer=(128,128), offset="(2,0,0)", to="(reshape-east)", height=18, depth=18, width=(5,5), caption="2D DS\\\\+SpAttn\\\\128"),
    to_connection("reshape", "b2d1"),

    # 2D Block 2: DS-ResBlock(128→128) + SpatialAttention
    to_ConvConvRelu("b2d2", s_filer=128, n_filer=(128,128), offset="(1,0,0)", to="(b2d1-east)", height=14, depth=14, width=(5,5), caption="2D DS\\\\+SpAttn\\\\128"),
    to_connection("b2d1", "b2d2"),

    # GAP + FC
    to_SoftMax("fc", s_filer=2, offset="(2,0,0)", to="(b2d2-east)", height=5, depth=5, width=2, caption="FC-2"),
    to_connection("b2d2", "fc"),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'ad_hybrid.tex')
    print("Generated ad_hybrid.tex — compile with: pdflatex ad_hybrid.tex")
