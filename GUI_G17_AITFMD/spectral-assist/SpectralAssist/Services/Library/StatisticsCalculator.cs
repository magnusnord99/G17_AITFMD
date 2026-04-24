using System.Collections.Generic;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Library;

public static class StatisticsCalculator
{
    //ToDO: might need work
    /// <summary>
    /// Computes per-class statistics from a ClassificationReport's PatchPredictions,
    /// Class index 0 is treated as background/negative and excluded from
    /// positive-class selection. If the model has only one class, it is returned as-is.
    /// </summary>
    /// <param name="report">The ClassificationReport to source the prediction values from.</param>
    /// <returns>A list of per-class statistics.</returns>
    public static List<ClassStatistics> Compute(ClassificationReport report)
    {
        var classCount = report.Classes.Count;
        var stats = new List<ClassStatistics>(classCount);
        if (classCount == 0 || report.Predictions.Count == 0) 
            return stats;

        var counts = new int[classCount];
        var above50 = new int[classCount];
        var above80 = new int[classCount];

        foreach (var pred in report.Predictions)
        {
            for (var c = 0; c < classCount && c < pred.Probabilities.Length; c++)
            {
                if (pred.Probabilities[c] >= 0.5f) above50[c]++;
                if (pred.Probabilities[c] >= 0.8f) above80[c]++;
            }
            
            var winner = pred.PredictedClass;
            if (winner >= 0 && winner < classCount) counts[winner]++;
        }

        var n = report.Predictions.Count;
        for (var c = 0; c < classCount; c++)
        {
            stats.Add(new ClassStatistics
            {
                ClassIndex = c,
                ClassName = report.Classes[c],
                PercentWinner = n > 0 ? (int)System.Math.Round(100.0 * counts[c] / n) : 0,
                PercentAbove50 = n > 0 ? 100.0 * above50[c] / n : 0,
                PercentAbove80 = n > 0 ? 100.0 * above80[c] / n : 0,
            });
        }

        return stats;
    }

    /// <summary>
    /// Returns the non-background class with the highest <c>PercentAbove50</c>.
    /// Class index 0 is skipped by default. Falls back to class 0 only when the
    /// model has a single class (nothing else to pick).
    /// </summary>
    public static (string Name, double Pct50, double Pct80) PickPositiveClass(List<ClassStatistics> stats)
    {
        if (stats.Count == 0) return ("", 0, 0);

        ClassStatistics? best = null;
        for (var i = 1; i < stats.Count; i++)
        {
            if (best == null || stats[i].PercentAbove50 > best.PercentAbove50)
                best = stats[i];
        }

        best ??= stats[0];
        return (best.ClassName, best.PercentAbove50, best.PercentAbove80);
    }
}