using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Hsi;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Flat-field calibration: reflectance = (raw - dark) / (white - dark + eps).
/// Matches Python <c>calibrate_cube</c> exactly (epsilon-based denominator).
///
/// This differs from <see cref="HsiCalibration.ApplyReflectance"/> which uses a
/// <c>d > 1 ? 1/d : 0</c> guard designed for raw sensor counts. For preprocessing
/// parity with Python, this eps-based version is used by the manifest-driven pipeline.
/// </summary>
public static class Calibration
{
    /// <summary>
    /// Applies reflectance calibration: (raw - dark) / (white - dark + eps).
    /// Each band is processed independently in parallel.
    /// </summary>
    public static HsiCube Apply(HsiCube raw, HsiCube dark, HsiCube white, float eps)
    {
        var bands = raw.Bands;
        var pixels = raw.PixelsPerBand;
        var result = new float[bands * pixels];

        Parallel.For(0, bands, b =>
        {
            var rawBand = raw.GetBand(b);
            var darkBand = dark.GetBand(b);
            var whiteBand = white.GetBand(b);
            var offset = b * pixels;

            for (var i = 0; i < pixels; i++)
            {
                var denom = whiteBand[i] - darkBand[i] + eps;
                result[offset + i] = (rawBand[i] - darkBand[i]) / denom;
            }
        });

        return new HsiCube(raw.Header, result);
    }
}