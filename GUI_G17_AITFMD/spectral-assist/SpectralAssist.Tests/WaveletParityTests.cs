using SpectralAssist.Models;
using SpectralAssist.Services.Preprocessing;
using SpectralAssist.Tests.Utils;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Parity with Python <c>wavelet.reduce_cube_wavelet_approx_padded</c>.
/// Regenerate fixtures: <c>ML_PIPELINE_G17_AITFMD/scripts/export_wavelet_golden.py</c>.
/// </summary>
public class WaveletParityTests
{
    private const float Tolerance = 1e-4f;

    [Fact]
    public void Approx_padded_db2_matches_python_golden()
    {
        var inputHwb = LoadHwbCube(4, 4, 275, "cube_hwb_f32.bin");
        var expectedHwb = LoadHwbCube(4, 4, 16, "wavelet16_expect.bin");

        // Convert HWB fixtures to BSQ HsiCube
        var inputBsq = FloatCubeToHsiCube.ToHsiCube(inputHwb);
        var expectedBsq = FloatCubeToHsiCube.ToHsiCube(expectedHwb);
        
        var actual = WaveletReducer.Apply(
            inputBsq,
            targetBands: 16,
            level: null,
            mode: "periodization",
            padMode: "edge");

        var diff = MaxAbsDiff(actual, expectedBsq);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }

    private static FloatCubeHWB LoadHwbCube(int lines, int samples, int bands, string fileName)
    {
        var baseDir = AppContext.BaseDirectory;
        var path = Path.Combine(baseDir, "Fixtures", "wavelet_golden", fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Wavelet golden fixture not found: {path}");

        var expected = lines * samples * bands * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");

        var floats = new float[lines * samples * bands];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return new FloatCubeHWB(lines, samples, bands, floats);
    }

    private static float MaxAbsDiff(HsiCube cubeA, HsiCube cubeB)
    {
        if (cubeA.Lines != cubeB.Lines || cubeA.Samples != cubeB.Samples || cubeA.Bands != cubeB.Bands)
            throw new ArgumentException("Shape mismatch.");
        var max = 0f;
        for (var band = 0; band < cubeA.Bands; band++)
        {
            var aBand = cubeA.GetBand(band);
            var bBand = cubeB.GetBand(band);
            for (var i = 0; i < aBand.Length; i++)
            {
                var d = Math.Abs(aBand[i] - bBand[i]);
                if (d > max) max = d;
            }
        }

        return max;
    }
}
