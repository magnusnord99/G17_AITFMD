using System;
using System.Text;

namespace SpectralAssist.Models;

/// <summary>Derived metrics from a <see cref="ClassificationReport"/>.</summary>
public static class ClassificationResultMetrics
{
    /// <summary>
    /// Share of patches whose softmax probability for <paramref name="classIndex"/> exceeds <paramref name="threshold"/>.
    /// Returns <c>null</c> if there are no predictions or any patch lacks that probability slot.
    /// </summary>
    public static double? FractionPatchesWithClassProbabilityOverThreshold(
        ClassificationReport result,
        int classIndex = 1,
        float threshold = 0.5f)
    {
        if (!TryCountPatchesWithClassProbabilityOverThreshold(result, out var over, out var total, classIndex, threshold))
            return null;
        return (double)over / total;
    }

    /// <summary>
    /// Counts patches with P(classIndex) &gt; threshold. Returns <c>false</c> if there are no predictions or any patch lacks that slot.
    /// </summary>
    public static bool TryCountPatchesWithClassProbabilityOverThreshold(
        ClassificationReport result,
        out int overCount,
        out int totalCount,
        int classIndex = 1,
        float threshold = 0.5f)
    {
        overCount = 0;
        totalCount = result.Predictions.Count;
        if (totalCount == 0)
            return false;

        foreach (var p in result.Predictions)
        {
            if (p.Probabilities.Length <= classIndex)
                return false;
            if (p.Probabilities[classIndex] > threshold)
                overCount++;
        }

        return true;
    }

    /// <summary>Human-readable summary for UI and PDF (no per-patch listing).</summary>
    public static string BuildReportSummaryText(ClassificationReport result, string manifestDisplayName = "")
    {
        var text = new StringBuilder();
        text.AppendLine($"Model: {result.ModelName}");
        if (!string.IsNullOrEmpty(manifestDisplayName))
            text.AppendLine($"Package: {manifestDisplayName}");

        text.AppendLine($"Evaluated: {result.EvaluatedPatches} patches ({result.SkippedPatches} skipped as background)");

        if (TryCountPatchesWithClassProbabilityOverThreshold(result, out var over, out var total))
        {
            var pct = 100.0 * over / total;
            text.AppendLine($"Patches with P(cancer) > 50 %: {pct:F1} % ({over} of {total})");
        }
        else
            text.AppendLine("Patches with P(cancer) > 50 %: —");

        return text.ToString();
    }
}
