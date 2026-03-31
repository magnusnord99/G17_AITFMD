using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Reduces band count by averaging variable-sized bins (crop or uneven strategy).
/// Matches Python <c>band_reduce._compute_bin_sizes</c>.
/// </summary>
public static class BandAverageReducer
{
    public static HsiCube Apply(HsiCube cube, int nOutBands, string strategy)
    {
        var bins = ComputeBinSizes(cube.Bands, nOutBands, strategy);
        var plane = cube.PixelsPerBand;
        var result = new float[nOutBands * plane];

        // Precompute source band offsets so the outer loop can run in parallel
        var srcBandStarts = new int[nOutBands];
        var cumulative = 0;
        for (var i = 0; i < nOutBands; i++)
        {
            srcBandStarts[i] = cumulative;
            cumulative += bins[i];
        }

        Parallel.For(0, nOutBands, outB =>
        {
            var size = bins[outB];
            var outOffset = outB * plane;
            var srcBandStart = srcBandStarts[outB];

            // Accumulate 'size' contiguous band planes
            for (var k = 0; k < size; k++)
            {
                var srcBand = cube.GetBand(srcBandStart + k);
                for (var i = 0; i < plane; i++)
                    result[outOffset + i] += srcBand[i];
            }

            // Divide by bin size
            var invSize = 1f / size;
            for (var i = 0; i < plane; i++)
                result[outOffset + i] *= invSize;
        });

        var header = new HsiHeader
        {
            Lines = cube.Lines,
            Samples = cube.Samples,
            Bands = nOutBands,
            Interleave = "bsq",
        };
        return new HsiCube(header, result);
    }

    /// <summary>Same logic as Python <c>_compute_bin_sizes</c>.</summary>
    private static int[] ComputeBinSizes(int nIn, int nOut, string strategy)
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