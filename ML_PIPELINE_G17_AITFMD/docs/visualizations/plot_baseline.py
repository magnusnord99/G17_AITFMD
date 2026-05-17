"""PlotNeuralNet diagram for Baseline3DCNN.

Baseline3DCNN: [Conv3D+GN+ReLU+Pool] x N → AdaptiveAvgPool → Dropout → FC
Default: channels=(32, 64), max_pool_layers=2
"""
import sys
sys.path.insert(0, '/Users/magnusnordmo/PlotNeuralNet')
from pycore.tikzeng import *

PLOTNN = '/Users/magnusnordmo/PlotNeuralNet'

arch = [
    to_head(PLOTNN),
    to_cor(),
    to_begin(),

    to_Conv("input",  s_filer=1,  n_filer=1,  offset="(0,0,0)",        to="(0,0,0)",        height=40, depth=40, width=1,  caption="Input"),
    to_Conv("conv1",  s_filer=32, n_filer=32, offset="(2,0,0)",        to="(input-east)",   height=40, depth=40, width=2,  caption="Conv3D-32"),
    to_connection("input", "conv1"),
    to_Pool("pool1",  offset="(0,0,0)",  to="(conv1-east)",  height=32, depth=32, width=1,  caption="Pool"),
    to_Conv("conv2",  s_filer=64, n_filer=64, offset="(1,0,0)",        to="(pool1-east)",   height=28, depth=28, width=4,  caption="Conv3D-64"),
    to_connection("pool1", "conv2"),
    to_Pool("pool2",  offset="(0,0,0)",  to="(conv2-east)",  height=20, depth=20, width=1,  caption="Pool"),
    to_SoftMax("fc",  s_filer=2,  offset="(2,0,0)",  to="(pool2-east)", height=6, depth=6,  width=2,  caption="FC-2"),
    to_connection("pool2", "fc"),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'baseline.tex')
    print("Generated baseline.tex — compile with: pdflatex baseline.tex")
