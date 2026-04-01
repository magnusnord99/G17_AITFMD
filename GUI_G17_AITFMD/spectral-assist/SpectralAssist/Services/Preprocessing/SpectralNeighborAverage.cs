using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Spectral band reduction by averaging adjacent bands — matches
/// <c>reduce_bands_neighbor_average</c> in <c>spectral_transform.py</c>.
/// </summary>
public static class SpectralNeighborAverage
{
    public static FloatCubeHWB Apply(FloatCubeHWB cube, int window)
    {
        if (window <= 1)
            return FloatCubeHWB.FromFlat(cube.Lines, cube.Samples, cube.Bands, cube.Data);

        var h = cube.Lines;
        var w = cube.Samples;
        var b = cube.Bands;
        var bReduced = b / window;
        if (bReduced == 0)
            throw new ArgumentException($"window={window} is too large for bands={b}.");

        var trimmedBands = bReduced * window;
        var src = cube.Data;
        var outLen = h * w * bReduced;
        var o = new float[outLen];

        for (var y = 0; y < h; y++)
        {
            for (var x = 0; x < w; x++)
            {
                for (var br = 0; br < bReduced; br++)
                {
                    double sum = 0;
                    for (var k = 0; k < window; k++)
                    {
                        var bi = br * window + k;
                        var idx = FloatCubeHWB.FlatIndex(y, x, bi, w, b);
                        sum += src[idx];
                    }

                    var oi = FloatCubeHWB.FlatIndex(y, x, br, w, bReduced);
                    o[oi] = (float)(sum / window);
                }
            }
        }

        return new FloatCubeHWB(h, w, bReduced, o);
    }
}
