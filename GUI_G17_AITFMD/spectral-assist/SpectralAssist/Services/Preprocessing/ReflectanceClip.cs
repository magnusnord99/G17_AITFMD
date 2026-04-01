using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Matches Python <c>np.clip(cube, clip_min, clip_max)</c> in <c>calibrateClip.clip_cube</c>.
/// </summary>
public static class ReflectanceClip
{
    public static FloatCubeHWB Apply(FloatCubeHWB cube, float clipMin, float clipMax)
    {
        var n = cube.Data.Length;
        var o = new float[n];
        var src = cube.Data;
        for (var i = 0; i < n; i++)
        {
            var v = src[i];
            if (v < clipMin) v = clipMin;
            else if (v > clipMax) v = clipMax;
            o[i] = v;
        }

        return new FloatCubeHWB(cube.Lines, cube.Samples, cube.Bands, o);
    }
}
