using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Parity with Python pipeline (see ML_PIPELINE_G17_AITFMD/scripts/export_baseline_golden.py).
/// Regenerate golden files after changing preprocessing math.
/// </summary>
public class BaselinePreprocessingParityTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Small_chain_through_avg3_matches_python_golden()
    {
        var opts = new BaselinePreprocessingOptions(
            calibrationEpsilon: 1e-8f,
            clipMin: 0f,
            clipMax: 1f,
            neighborAverageWindow: 3);

        var raw = GoldenFloatLoader.LoadCube(4, 4, 9, "small_raw.bin");
        var dark = GoldenFloatLoader.LoadCube(4, 4, 9, "small_dark.bin");
        var white = GoldenFloatLoader.LoadCube(4, 4, 9, "small_white.bin");
        var expected = GoldenFloatLoader.LoadCube(4, 4, 3, "small_expect_after_avg3.bin");

        var actual = BaselineSpectralPipeline.RunThroughAvg3(raw, dark, white, opts);
        var diff = GoldenFloatLoader.MaxAbsDiff(actual, expected);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }

    [Fact]
    public void Chain275_through_avg16_matches_python_golden()
    {
        var opts = new BaselinePreprocessingOptions(
            calibrationEpsilon: 1e-8f,
            clipMin: 0f,
            clipMax: 1f,
            neighborAverageWindow: 3,
            bandReduceOutBands: 16,
            bandReduceStrategy: "crop");

        var raw = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_raw.bin");
        var dark = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_dark.bin");
        var white = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_white.bin");
        var expected = GoldenFloatLoader.LoadCube(4, 4, 16, "chain275_expect_avg16.bin");

        var actual = BaselineSpectralPipeline.RunThroughAvg16(raw, dark, white, opts);
        var diff = GoldenFloatLoader.MaxAbsDiff(actual, expected);
        Assert.True(diff < Tolerance, $"max abs diff {diff} >= {Tolerance}");
    }
}
