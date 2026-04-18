using SpectralAssist.Models;
using Xunit;

namespace SpectralAssist.Tests;

public class ClassificationResultMetricsTests
{
    [Fact]
    public void Fraction_ReturnsNull_WhenNoPredictions()
    {
        var r = new ClassificationResult { Predictions = [] };
        Assert.Null(ClassificationResultMetrics.FractionPatchesWithClassProbabilityOverThreshold(r));
    }

    [Fact]
    public void Fraction_ReturnsNull_WhenClassIndexMissing()
    {
        var r = new ClassificationResult
        {
            Predictions =
            [
                new PatchPrediction { Probabilities = [1f] },
            ],
        };
        Assert.Null(ClassificationResultMetrics.FractionPatchesWithClassProbabilityOverThreshold(r, classIndex: 1));
    }

    [Fact]
    public void Fraction_AllOverThreshold_ReturnsOne()
    {
        var r = new ClassificationResult
        {
            Predictions =
            [
                new PatchPrediction { Probabilities = [0f, 0.9f] },
                new PatchPrediction { Probabilities = [0f, 0.6f] },
            ],
        };
        var f = ClassificationResultMetrics.FractionPatchesWithClassProbabilityOverThreshold(r);
        Assert.Equal(1.0, f);
    }

    [Fact]
    public void Fraction_Mixed_CountsCorrectly()
    {
        var r = new ClassificationResult
        {
            Predictions =
            [
                new PatchPrediction { Probabilities = [0.5f, 0.51f] },
                new PatchPrediction { Probabilities = [0.2f, 0.55f] },
                new PatchPrediction { Probabilities = [0.1f, 0.49f] },
            ],
        };
        Assert.True(ClassificationResultMetrics.TryCountPatchesWithClassProbabilityOverThreshold(
            r, out var over, out var total));
        Assert.Equal(2, over);
        Assert.Equal(3, total);
        Assert.Equal(2.0 / 3.0, ClassificationResultMetrics.FractionPatchesWithClassProbabilityOverThreshold(r));
    }

    [Fact]
    public void TryCount_ThresholdIsStrictlyGreaterThan()
    {
        var r = new ClassificationResult
        {
            Predictions = [new PatchPrediction { Probabilities = [0f, 0.5f] }],
        };
        Assert.True(ClassificationResultMetrics.TryCountPatchesWithClassProbabilityOverThreshold(
            r, out var over, out _, threshold: 0.5f));
        Assert.Equal(0, over);
    }
}
