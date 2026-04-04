using System;
using System.IO;
using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Parity with Python <c>tissue_mask.build_tissue_mask</c> (<c>mean_std_percentile</c>).
/// Regenerate fixtures: <c>ML_PIPELINE_G17_AITFMD/scripts/export_tissue_mask_golden.py</c>.
/// </summary>
public class TissueMaskParityTests
{
    [Fact]
    public void Mean_std_percentile_mask_matches_python_golden()
    {
        var cube = GoldenFloatLoader.LoadCubeFromSubDir(16, 16, 16, "tissue_mask_golden", "cube_hwb_f32.bin");
        var expectedMask = LoadMaskBytes(16, 16, "tissue_mask_golden", "mask_uint8.bin");

        var opts = new TissueMaskOptions(
            qMean: 0.5f,
            qStd: 0.4f,
            minObjectSize: 4,
            minHoleSize: 4);

        var actual = TissueMask.BuildMask(cube, opts);
        Assert.Equal(expectedMask.Length, actual.Length);
        for (var i = 0; i < actual.Length; i++)
        {
            var e = expectedMask[i];
            var a = actual[i] ? (byte)1 : (byte)0;
            Assert.True(e == a, $"mask diff at {i}: expected {e}, got {a}");
        }
    }

    private static byte[] LoadMaskBytes(int lines, int samples, string subDir, string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", subDir, fileName);
        var n = lines * samples;
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != n)
            throw new InvalidDataException($"Expected {n} bytes in {fileName}, got {bytes.Length}.");
        return bytes;
    }
}