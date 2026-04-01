using System;
using System.Collections.Generic;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Reduce band count by averaging groups — matches <c>reduce_bands_by_avg</c> in <c>band_reduce.py</c>.
/// </summary>
public static class BandAverageReducer
{
    public static FloatCubeHWB Apply(FloatCubeHWB cube, int nOutBands, string strategy)
    {
        var bins = ComputeBinSizes(cube.Bands, nOutBands, strategy);
        var h = cube.Lines;
        var w = cube.Samples;
        var src = cube.Data;
        var o = new float[h * w * nOutBands];

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                var start = 0;
                for (var outB = 0; outB < nOutBands; outB++)
                {
                    var size = bins[outB];
                    double sum = 0;
                    for (var k = 0; k < size; k++)
                    {
                        var bi = start + k;
                        var idx = FloatCubeHWB.FlatIndex(y, x, bi, w, cube.Bands);
                        sum += src[idx];
                    }

                    var oIdx = FloatCubeHWB.FlatIndex(y, x, outB, w, nOutBands);
                    o[oIdx] = (float)(sum / size);
                    start += size;
                }
            }
        }

        return new FloatCubeHWB(h, w, nOutBands, o);
    }

    /// <summary>Same logic as Python <c>_compute_bin_sizes</c>.</summary>
    public static IReadOnlyList<int> ComputeBinSizes(int nIn, int nOut, string strategy)
    {
        if (strategy == "crop")
        {
            var nUsable = nIn / nOut * nOut;
            if (nUsable == 0)
                throw new ArgumentException($"Cannot crop {nIn} bands to {nOut} — too few bands.");
            var baseSize = nUsable / nOut;
            var list = new int[nOut];
            for (var i = 0; i < nOut; i++)
                list[i] = baseSize;
            return list;
        }

        if (strategy == "uneven")
        {
            var baseSize = nIn / nOut;
            var remainder = nIn % nOut;
            var list = new int[nOut];
            for (var i = 0; i < remainder; i++)
                list[i] = baseSize + 1;
            for (var i = remainder; i < nOut; i++)
                list[i] = baseSize;
            return list;
        }

        throw new ArgumentException($"Unknown strategy '{strategy}'. Use 'crop' or 'uneven'.");
    }
}
