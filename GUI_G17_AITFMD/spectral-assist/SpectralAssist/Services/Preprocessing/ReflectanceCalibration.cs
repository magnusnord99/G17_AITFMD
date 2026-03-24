using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Flat-field calibration matching Python:
/// <c>calibrated = (raw - dark) / (white - dark + eps)</c> in <c>calibrateClip.calibrate_cube</c>.
/// </summary>
public static class ReflectanceCalibration
{
    public static FloatCubeHWB Apply(FloatCubeHWB raw, FloatCubeHWB dark, FloatCubeHWB white, float eps)
    {
        if (raw.Bands != dark.Bands || raw.Bands != white.Bands)
            throw new ArgumentException("raw/dark/white band count mismatch.");

        var fullFrame = raw.Lines == dark.Lines && raw.Samples == dark.Samples
                        && raw.Lines == white.Lines && raw.Samples == white.Samples;

        if (fullFrame)
            return ApplyFullFrame(raw, dark, white, eps);

        // Match <see cref="HsiCalibration.ApplyReflectance"/>: when reference cubes have fewer
        // lines than the scene (e.g. single scan-line dark/white), broadcast row 0 across all lines.
        var lineRef = dark.Lines != raw.Lines
                      && raw.Samples == dark.Samples && raw.Samples == white.Samples
                      && dark.Lines >= 1 && white.Lines >= 1;

        if (lineRef)
            return ApplyLineReferenceRow0(raw, dark, white, eps);

        throw new ArgumentException("raw/dark shape mismatch.");
    }

    private static FloatCubeHWB ApplyFullFrame(FloatCubeHWB raw, FloatCubeHWB dark, FloatCubeHWB white, float eps)
    {
        var n = raw.Data.Length;
        var outData = new float[n];
        var rd = raw.Data;
        var dk = dark.Data;
        var wh = white.Data;

        for (var i = 0; i < n; i++)
        {
            var denom = wh[i] - dk[i] + eps;
            outData[i] = (rd[i] - dk[i]) / denom;
        }

        return new FloatCubeHWB(raw.Lines, raw.Samples, raw.Bands, outData);
    }

    /// <summary>
    /// Uses dark/white at line index 0 for every raw line (same as HSI display calibration).
    /// </summary>
    private static FloatCubeHWB ApplyLineReferenceRow0(
        FloatCubeHWB raw,
        FloatCubeHWB dark,
        FloatCubeHWB white,
        float eps)
    {
        var lines = raw.Lines;
        var samples = raw.Samples;
        var bands = raw.Bands;
        var outData = new float[raw.Data.Length];
        var rd = raw.Data;
        var dk = dark.Data;
        var wh = white.Data;

        for (var y = 0; y < lines; y++)
        {
            for (var x = 0; x < samples; x++)
            {
                for (var b = 0; b < bands; b++)
                {
                    var iR = FloatCubeHWB.FlatIndex(y, x, b, samples, bands);
                    var iD = FloatCubeHWB.FlatIndex(0, x, b, samples, bands);
                    var iW = FloatCubeHWB.FlatIndex(0, x, b, samples, bands);
                    var denom = wh[iW] - dk[iD] + eps;
                    outData[iR] = (rd[iR] - dk[iD]) / denom;
                }
            }
        }

        return new FloatCubeHWB(lines, samples, bands, outData);
    }
}
