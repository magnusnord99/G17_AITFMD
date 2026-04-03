using SpectralAssist.Services.Preprocessing;
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
        var cube = LoadCube(4, 4, 275, "cube_hwb_f32.bin");
        var expected = LoadCube(4, 4, 16, "wavelet16_expect.bin");

        var actual = WaveletReducer.ApplyApproxPaddedDb2(
            cube,
            targetBands: 16,
            level: null,
            mode: "periodization",
            padMode: "edge");

        var diff = GoldenFloatLoader.MaxAbsDiff(actual, expected);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }

    private static FloatCubeHWB LoadCube(int lines, int samples, int bands, string fileName)
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
}
