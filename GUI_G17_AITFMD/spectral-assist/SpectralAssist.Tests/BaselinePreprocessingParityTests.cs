using SpectralAssist.Models;
using SpectralAssist.Services;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Parity with Python pipeline using golden files (see ML_PIPELINE_G17_AITFMD/scripts/export_baseline_golden.py).
/// Tests run the BSQ pipeline and compare against Python-generated expected outputs.
/// Regenerate golden files after changing preprocessing math.
/// </summary>
public class BaselinePreprocessingParityTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Small_chain_through_avg3_matches_python_golden()
    {
        // Steps: calibrate → clip → avg3 (no tissue mask, no band_average)
        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average"],
            Params = new PreprocessingConfig
            {
                CalibrationEpsilon = 1e-8f,
                ClipMin = 0f,
                ClipMax = 1f,
                NeighborAverageWindow = 3,
            }
        };

        var raw = GoldenFloatLoader.LoadCube(4, 4, 9, "small_raw.bin");
        var dark = GoldenFloatLoader.LoadCube(4, 4, 9, "small_dark.bin");
        var white = GoldenFloatLoader.LoadCube(4, 4, 9, "small_white.bin");
        var expected = GoldenFloatLoader.LoadCube(4, 4, 3, "small_expect_after_avg3.bin");

        var result = PreprocessingService.Run(raw, dark, white, preprocessing);
        var diff = GoldenFloatLoader.MaxAbsDiff(result.Cube, expected);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }

    [Fact]
    public void Chain275_through_avg16_matches_python_golden()
    {
        // Steps: calibrate → clip → avg3 → band_average (no tissue mask)
        var preprocessing = new PreprocessingInfo
        {
            Steps = ["calibrate", "clip", "neighbor_average", "band_average"],
            Params = new PreprocessingConfig
            {
                CalibrationEpsilon = 1e-8f,
                ClipMin = 0f,
                ClipMax = 1f,
                NeighborAverageWindow = 3,
                BandReduceOutBands = 16,
                BandReduceStrategy = "crop",
            }
        };

        var raw = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_raw.bin");
        var dark = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_dark.bin");
        var white = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_white.bin");
        var expected = GoldenFloatLoader.LoadCube(4, 4, 16, "chain275_expect_avg16.bin");

        var result = PreprocessingService.Run(raw, dark, white, preprocessing);
        var diff = GoldenFloatLoader.MaxAbsDiff(result.Cube, expected);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }
}