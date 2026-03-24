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
    private const float MapTolerance = 1e-5f;

    [Fact]
    public void Mean_and_std_maps_match_python_golden()
    {
        var cube = LoadCube(16, 16, 16, "tissue_mask_golden", "cube_hwb_f32.bin");
        var expectedMean = LoadMap(16, 16, "tissue_mask_golden", "mean_map_f32.bin");
        var expectedStd = LoadMap(16, 16, "tissue_mask_golden", "std_map_f32.bin");

        var mean = TissueMaskMeanStdPercentile.ComputeMeanMap(cube);
        var std = TissueMaskMeanStdPercentile.ComputeStdMap(cube);

        Assert.Equal(expectedMean.Length, mean.Length);
        Assert.Equal(expectedStd.Length, std.Length);
        Assert.True(MaxAbsDiff(expectedMean, mean) < MapTolerance);
        Assert.True(MaxAbsDiff(expectedStd, std) < MapTolerance);
    }

    [Fact]
    public void Mean_std_percentile_mask_matches_python_golden()
    {
        var cube = LoadCube(16, 16, 16, "tissue_mask_golden", "cube_hwb_f32.bin");
        var expectedMask = LoadMaskBytes(16, 16, "tissue_mask_golden", "mask_uint8.bin");

        var opts = new TissueMaskMeanStdOptions(
            qMean: 0.5f,
            qStd: 0.4f,
            minObjectSize: 4,
            minHoleSize: 4);

        var actual = TissueMaskMeanStdPercentile.BuildMask(cube, opts);
        Assert.Equal(expectedMask.Length, actual.Length);
        for (var i = 0; i < actual.Length; i++)
        {
            var e = expectedMask[i];
            var a = actual[i] ? (byte)1 : (byte)0;
            Assert.True(e == a, $"mask diff at {i}: expected {e}, got {a}");
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        var max = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            var d = Math.Abs(a[i] - b[i]);
            if (d > max) max = d;
        }

        return max;
    }

    private static FloatCubeHWB LoadCube(int lines, int samples, int bands, string subDir, string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", subDir, fileName);
        var expected = lines * samples * bands * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");
        var floats = new float[lines * samples * bands];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return new FloatCubeHWB(lines, samples, bands, floats);
    }

    private static float[] LoadMap(int lines, int samples, string subDir, string fileName)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", subDir, fileName);
        var n = lines * samples;
        var expected = n * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");
        var floats = new float[n];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
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
