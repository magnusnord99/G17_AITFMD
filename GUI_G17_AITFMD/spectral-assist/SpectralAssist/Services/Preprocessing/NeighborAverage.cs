using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Reduces band count by averaging groups of <c>window</c> contiguous bands.
/// e.g. 825 bands with window 3 produces 275 bands.
/// </summary>
public static class NeighborAverage
{
    public static HsiCube Apply(HsiCube cube, int window)
    {
        var bReduced = cube.Bands / window;
        var plane = cube.PixelsPerBand;
        var result = new float[bReduced * plane];

        Parallel.For(0, bReduced, outB =>
        {
            var outOffset = outB * plane;

            // Accumulate "window" contiguous band planes
            for (var k = 0; k < window; k++)
            {
                var srcBand = cube.GetBand(outB * window + k);
                for (var i = 0; i < plane; i++)
                    result[outOffset + i] += srcBand[i];
            }

            // Divide by window
            var invW = 1f / window;
            for (var i = 0; i < plane; i++)
                result[outOffset + i] *= invW;
        });

        var header = new HsiHeader
        {
            Lines = cube.Lines,
            Samples = cube.Samples,
            Bands = bReduced,
            Interleave = "bsq",
        };
        return new HsiCube(header, result);
    }
}