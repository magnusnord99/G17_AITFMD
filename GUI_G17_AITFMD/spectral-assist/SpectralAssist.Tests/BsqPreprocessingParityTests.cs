using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Standalone unit tests for each BSQ preprocessing step.
/// Verifies correctness with synthetic data (deterministic, no golden files needed).
/// </summary>
public class BsqPreprocessingParityTests
{
    [Fact]
    public void Clip_clamps_values_to_range()
    {
        const int h = 2, w = 2, b = 2;
        // Values: 0.0, 0.3, 0.6, 0.9 per band
        var data = new float[] { 0.0f, 0.3f, 0.6f, 0.9f, 0.0f, 0.3f, 0.6f, 0.9f };
        var cube = MakeBsqCube(h, w, b, data);

        ReflectanceClip.ApplyInPlace(cube, 0.1f, 0.7f);

        var band0 = cube.GetBand(0);
        Assert.Equal(0.1f, band0[0]); // 0.0 clipped up
        Assert.Equal(0.3f, band0[1]); // in range
        Assert.Equal(0.6f, band0[2]); // in range
        Assert.Equal(0.7f, band0[3]); // 0.9 clipped down
    }

    [Fact]
    public void Neighbor_average_reduces_bands_by_window()
    {
        const int h = 2, w = 2, b = 6, window = 3;
        var cube = MakeSequentialBsqCube(h, w, b);

        var result = NeighborAverage.Apply(cube, window);

        Assert.Equal(2, result.Bands); // 6 / 3 = 2
        Assert.Equal(h, result.Lines);
        Assert.Equal(w, result.Samples);
    }

    [Fact]
    public void Neighbor_average_computes_correct_mean()
    {
        // 1x1 image, 3 bands with values [1, 2, 3], window=3 → result = [2.0]
        var data = new float[] { 1f, 2f, 3f };
        var cube = MakeBsqCube(1, 1, 3, data);

        var result = NeighborAverage.Apply(cube, 3);

        Assert.Equal(1, result.Bands);
        Assert.Equal(2f, result.GetBand(0)[0]);
    }

    [Fact]
    public void Band_average_crop_discards_remainder()
    {
        const int h = 2, w = 2, b = 7, outBands = 3;
        var cube = MakeSequentialBsqCube(h, w, b);

        var result = BandAverageReducer.Apply(cube, outBands, "crop");

        // crop: 7 / 3 = 2 usable per bin (6 total), 1 band discarded
        Assert.Equal(outBands, result.Bands);
    }

    [Fact]
    public void Band_average_uneven_uses_all_bands()
    {
        const int h = 2, w = 2, b = 7, outBands = 3;
        var cube = MakeSequentialBsqCube(h, w, b);

        var result = BandAverageReducer.Apply(cube, outBands, "uneven");

        // uneven: 7 / 3 = 2 base, remainder 1 → bins: [3, 2, 2] = 7 total (all used)
        Assert.Equal(outBands, result.Bands);
    }

    [Fact]
    public void Tissue_mask_returns_correct_size()
    {
        const int h = 8, w = 8, b = 4;
        var cube = MakeRandomBsqCube(h, w, b, seed: 42);
        var mask = TissueMask.BuildMask(cube, qMean: 0.5f, qStd: 0.4f, minObjectSize: 1, minHoleSize: 1);

        Assert.Equal(h * w, mask.Length);
    }

    [Fact]
    public void Calibration_matches_expected_formula()
    {
        // raw=10, dark=2, white=5, eps=0 → (10-2)/(5-2+0) = 8/3 ≈ 2.6667
        var raw = MakeBsqCube(1, 1, 1, [10f]);
        var dark = MakeBsqCube(1, 1, 1, [2f]);
        var white = MakeBsqCube(1, 1, 1, [5f]);

        var result = Calibration.Apply(raw, dark, white, eps: 0f);

        Assert.Equal(8f / 3f, result.GetBand(0)[0], precision: 5);
    }

    [Fact]
    public void Full_pipeline_reduces_bands_and_produces_mask()
    {
        const int h = 4, w = 4, bandsIn = 9;
        var raw = MakeRandomBsqCube(h, w, bandsIn, seed: 42);
        var dark = MakeZeroBsqCube(h, w, bandsIn);
        var white = MakeOnesBsqCube(h, w, bandsIn);

        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"],
            Params = new PreprocessingConfig
            {
                CalibrationEpsilon = 1e-8f,
                ClipMin = 0f,
                ClipMax = 1f,
                NeighborAverageWindow = 3,
                BandReduceOutBands = 3,
                BandReduceStrategy = "uneven",
                TissueMaskQMean = 0.5f,
                TissueMaskQStd = 0.4f,
                TissueMaskMinObjectSize = 1,
                TissueMaskMinHoleSize = 1,
                TissueMaskMethod = "mean_std_percentile"
            }
        };

        var result = PreprocessingService.Run(raw, dark, white, preprocessing);

        Assert.Equal(3, result.Cube.Bands);
        Assert.Equal(h, result.Cube.Lines);
        Assert.Equal(w, result.Cube.Samples);
        Assert.NotNull(result.TissueMask);
        Assert.Equal(h * w, result.TissueMask!.Length);
    }

    // --- Helpers --- //

    private static HsiCube MakeBsqCube(int h, int w, int b, float[] data)
    {
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }

    private static HsiCube MakeSequentialBsqCube(int h, int w, int b)
    {
        var data = new float[b * h * w];
        for (var i = 0; i < data.Length; i++)
            data[i] = i;
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }

    private static HsiCube MakeRandomBsqCube(int h, int w, int b, int seed)
    {
        var rng = new Random(seed);
        var data = new float[b * h * w];
        for (var i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 0.5);
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }

    private static HsiCube MakeZeroBsqCube(int h, int w, int b)
    {
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, new float[b * h * w]);
    }

    private static HsiCube MakeOnesBsqCube(int h, int w, int b)
    {
        var data = new float[b * h * w];
        Array.Fill(data, 1f);
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }
}