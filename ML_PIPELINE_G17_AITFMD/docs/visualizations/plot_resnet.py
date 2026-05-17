"""PlotNeuralNet diagram for ResNet3DCNN.

ResNet3DCNN: Entry conv+pool → [ResBlock x N + Pool] per stage → GAP → FC
Default: stage_channels=(16,32,64), num_blocks=(2,2,2)
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

    # Entry: Conv3D(1→16) + GN + ReLU + MaxPool
    to_Conv("entry", s_filer=16, n_filer=16, offset="(2,0,0)", to="(input-east)", height=40, depth=40, width=2, caption="Entry\\\\Conv-16"),
    to_connection("input", "entry"),
    to_Pool("epool", offset="(0,0,0)", to="(entry-east)", height=32, depth=32, width=1, caption=""),

    # Stage 1: 2x ResBlock(16→16) + Pool
    to_ConvConvRelu("s1", s_filer=16, n_filer=(16,16), offset="(1,0,0)", to="(epool-east)", height=32, depth=32, width=(2,2), caption="Res x2\\\\16"),
    to_connection("epool", "s1"),
    to_Pool("p1", offset="(0,0,0)", to="(s1-east)", height=24, depth=24, width=1, caption=""),

    # Stage 2: 2x ResBlock(16→32) + Pool
    to_ConvConvRelu("s2", s_filer=32, n_filer=(32,32), offset="(1,0,0)", to="(p1-east)", height=24, depth=24, width=(3,3), caption="Res x2\\\\32"),
    to_connection("p1", "s2"),
    to_Pool("p2", offset="(0,0,0)", to="(s2-east)", height=16, depth=16, width=1, caption=""),

    # Stage 3: 2x ResBlock(32→64) + Pool
    to_ConvConvRelu("s3", s_filer=64, n_filer=(64,64), offset="(1,0,0)", to="(p2-east)", height=16, depth=16, width=(4,4), caption="Res x2\\\\64"),
    to_connection("p2", "s3"),
    to_Pool("p3", offset="(0,0,0)", to="(s3-east)", height=10, depth=10, width=1, caption=""),

    # GAP + FC
    to_SoftMax("fc", s_filer=2, offset="(2,0,0)", to="(p3-east)", height=5, depth=5, width=2, caption="FC-2"),
    to_connection("p3", "fc"),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'resnet.tex')
    print("Generated resnet.tex — compile with: pdflatex resnet.tex")
