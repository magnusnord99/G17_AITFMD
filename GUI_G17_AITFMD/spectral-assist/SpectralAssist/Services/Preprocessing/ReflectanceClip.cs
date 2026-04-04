using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Clips reflectance values in-place to [clipMin, clipMax], parallelized by band.
/// Each band is a contiguous memory slice so threads don't contend on the same cache lines.
/// </summary>
public static class ReflectanceClip
{
    public static void ApplyInPlace(HsiCube cube, float clipMin, float clipMax)
    {
        Parallel.For(0, cube.Bands, b =>
        {
            var band = cube.GetBand(b);
            for (var i = 0; i < band.Length; i++)
            {
                if (band[i] < clipMin) band[i] = clipMin;
                else if (band[i] > clipMax) band[i] = clipMax;
            }
        });
    }
}