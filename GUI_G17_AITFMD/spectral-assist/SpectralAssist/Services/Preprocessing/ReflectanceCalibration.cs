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
        if (raw.Lines != dark.Lines || raw.Samples != dark.Samples || raw.Bands != dark.Bands)
            throw new ArgumentException("raw/dark shape mismatch.");
        if (raw.Lines != white.Lines || raw.Samples != white.Samples || raw.Bands != white.Bands)
            throw new ArgumentException("raw/white shape mismatch.");

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
}
