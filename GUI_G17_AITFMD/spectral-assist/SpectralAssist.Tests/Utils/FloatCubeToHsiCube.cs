using SpectralAssist.Models;

namespace SpectralAssist.Tests.Utils;

/// <summary>
/// Converts float cubes from pipeline layout (HWB, C-order like NumPy) to BSQ layout used by
/// <see cref="HsiCube"/>. Use <see cref="OnnxClassifier"/> for 4D [1,C,H,W] eller
/// <see cref="Onnx3DCnnClassifier"/> for 5D [1,1,C,H,W] (3D-CNN).
/// </summary>
public static class FloatCubeToHsiCube
{
    /// <summary>Build an <see cref="HsiCube"/> with BSQ storage from <see cref="FloatCubeHWB"/>.</summary>
    public static HsiCube ToHsiCube(FloatCubeHWB cube)
    {
        var header = new HsiHeader
        {
            Lines = cube.Lines,
            Samples = cube.Samples,
            Bands = cube.Bands,
            Interleave = "bsq",
        };
        return new HsiCube(header, ConvertHwbToBsq(cube));
    }

    /// <summary>
    /// HWB (line, sample, band) → BSQ: band-major, each band is row-major H×W.
    /// </summary>
    public static float[] ConvertHwbToBsq(FloatCubeHWB cube)
    {
        var h = cube.Lines;
        var w = cube.Samples;
        var b = cube.Bands;
        var plane = h * w;
        var dst = new float[b * plane];
        for (var band = 0; band < b; band++)
        {
            var bo = band * plane;
            for (var line = 0; line < h; line++)
            {
                for (var s = 0; s < w; s++)
                    dst[bo + line * w + s] = cube.Get(line, s, band);
            }
        }

        return dst;
    }
}
