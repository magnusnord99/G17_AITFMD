"""PlotNeuralNet diagram for MSDDenseCNN3D.

MSDDenseCNN3D: Stem → 3 parallel DenseBlock branches (different kernels)
               → concat → post-fusion conv x2 → GAP → FC
Default: stem=16ch, branches each grow to 32ch (16 + 2*8), post_fusion=96ch
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

    # Stem: Conv3D(1→16) + GN + ReLU + MaxPool
    to_Conv("stem", s_filer=16, n_filer=16, offset="(2,0,0)", to="(input-east)", height=40, depth=40, width=2, caption="Stem\\\\Conv-16"),
    to_connection("input", "stem"),
    to_Pool("stempool", offset="(0,0,0)", to="(stem-east)", height=32, depth=32, width=1, caption=""),

    # Branch 1 (kernel 1x3x3) — placed above centre
    to_Conv("b1", s_filer=32, n_filer=32, offset="(2,8,0)", to="(stempool-east)", height=28, depth=28, width=3, caption="Dense\\\\1x3x3"),
    to_connection("stempool", "b1"),

    # Branch 2 (kernel 3x3x3) — centre
    to_Conv("b2", s_filer=32, n_filer=32, offset="(2,0,0)", to="(stempool-east)", height=28, depth=28, width=3, caption="Dense\\\\3x3x3"),
    to_connection("stempool", "b2"),

    # Branch 3 (kernel 3x5x5) — below centre
    to_Conv("b3", s_filer=32, n_filer=32, offset="(2,-8,0)", to="(stempool-east)", height=28, depth=28, width=3, caption="Dense\\\\3x5x5"),
    to_connection("stempool", "b3"),

    # Sum node to represent concat
    to_Sum("concat", offset="(4,0,0)", to="(b2-east)", radius=3, opacity=0.6),
    to_connection("b1", "concat"),
    to_connection("b2", "concat"),
    to_connection("b3", "concat"),

    # Post-fusion: 1x1 conv then 3x3 conv → 96ch each
    to_Conv("pf1", s_filer=96, n_filer=96, offset="(2,0,0)", to="(concat-east)", height=20, depth=20, width=2, caption="Conv 1x1\\\\96"),
    to_connection("concat", "pf1"),
    to_Conv("pf2", s_filer=96, n_filer=96, offset="(0,0,0)", to="(pf1-east)", height=20, depth=20, width=2, caption="Conv 3x3\\\\96"),

    # GAP + FC
    to_SoftMax("fc", s_filer=2, offset="(2,0,0)", to="(pf2-east)", height=5, depth=5, width=2, caption="FC-2"),
    to_connection("pf2", "fc"),

    to_end()
]

if __name__ == '__main__':
    to_generate(arch, 'msd_dense.tex')
    print("Generated msd_dense.tex — compile with: pdflatex msd_dense.tex")
