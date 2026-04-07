using SpectralAssist.Models;
using SpectralAssist.Services;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Tests for <see cref="PreprocessingService"/>: manifest-driven step execution.
/// </summary>
public class PreprocessingServiceTests
{
    [Fact]
    public void Default_steps_produce_reduced_bands_and_mask()
    {
        const int h = 4, w = 4, bandsIn = 9;
        var sceneCube = MakeBsqCube(h, w, bandsIn, seed: 42);
        var darkCube = MakeZeroBsqCube(h, w, bandsIn);
        var whiteCube = MakeOnesBsqCube(h, w, bandsIn);

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
            }
        };

        var result = PreprocessingService.Run(sceneCube, darkCube, whiteCube, preprocessing);

        // avg3: 9/3 = 3 bands, then band_average with 3 out = 1 band each
        Assert.Equal(h, result.Cube.Lines);
        Assert.Equal(w, result.Cube.Samples);
        Assert.Equal(3, result.Cube.Bands);
        Assert.NotNull(result.TissueMask);
        Assert.Equal(h * w, result.TissueMask!.Length);
    }

    [Fact]
    public void Custom_steps_skip_tissue_mask()
    {
        const int h = 4, w = 4, bandsIn = 9;
        var raw = MakeBsqCube(h, w, bandsIn, seed: 42);
        var dark = MakeZeroBsqCube(h, w, bandsIn);
        var white = MakeOnesBsqCube(h, w, bandsIn);

        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average", "band_average"],
            Params = new PreprocessingConfig
            {
                NeighborAverageWindow = 3,
                BandReduceOutBands = 3,
                BandReduceStrategy = "uneven",
            }
        };

        var result = PreprocessingService.Run(raw, dark, white, preprocessing);

        Assert.Equal(3, result.Cube.Bands);
        Assert.Null(result.TissueMask);
    }

    [Fact]
    public void Unknown_step_throws()
    {
        var cube = MakeZeroBsqCube(2, 2, 3);
        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "magic_filter"],
            Params = new PreprocessingConfig()
        };

        Assert.Throws<NotSupportedException>(() =>
            PreprocessingService.Run(cube, cube, cube, preprocessing));
    }

    [Fact]
    public void RunFromCalibrated_skips_calibrate_step()
    {
        const int h = 4, w = 4, bandsIn = 9;
        // Pre-calibrated cube (values already in 0-1 range)
        var calibrated = MakeBsqCube(h, w, bandsIn, seed: 42);

        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"],
            Params = new PreprocessingConfig
            {
                ClipMin = 0f,
                ClipMax = 1f,
                NeighborAverageWindow = 3,
                BandReduceOutBands = 3,
                BandReduceStrategy = "uneven",
                TissueMaskQMean = 0.5f,
                TissueMaskQStd = 0.4f,
                TissueMaskMinObjectSize = 1,
                TissueMaskMinHoleSize = 1,
            }
        };

        var result = PreprocessingService.RunFromCalibrated(calibrated, preprocessing);

        Assert.Equal(3, result.Cube.Bands);
        Assert.NotNull(result.TissueMask);
    }

    [Fact]
    public void RunFromCalibrated_does_not_mutate_input()
    {
        const int h = 4, w = 4, bandsIn = 9;
        var calibrated = MakeBsqCube(h, w, bandsIn, seed: 42);

        // Save original first band values
        var originalBand0 = new float[h * w];
        calibrated.GetBand(0).CopyTo(originalBand0);

        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"],
            Params = new PreprocessingConfig
            {
                ClipMin = 0.1f,
                ClipMax = 0.4f,
                NeighborAverageWindow = 3,
                BandReduceOutBands = 3,
                BandReduceStrategy = "uneven",
                TissueMaskQMean = 0.5f,
                TissueMaskQStd = 0.4f,
                TissueMaskMinObjectSize = 1,
                TissueMaskMinHoleSize = 1,
            }
        };

        _ = PreprocessingService.RunFromCalibrated(calibrated, preprocessing);

        // Original cube must be unchanged
        var afterBand0 = calibrated.GetBand(0);
        for (var i = 0; i < originalBand0.Length; i++)
            Assert.Equal(originalBand0[i], afterBand0[i]);
    }


    // -- Helpers -- //

    private static HsiCube MakeBsqCube(int h, int w, int b, int seed)
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