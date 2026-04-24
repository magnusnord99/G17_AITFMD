using System.Linq;
using SpectralAssist.Models;

namespace SpectralAssist.Extensions;

public static class ImageNodeExtensions
{
    extension(ImageNode image)
    {
        public RunSummary? MostSevereRun() =>
            image.Runs.Count == 0 ? null : image.Runs.MaxBy(r => r.PositiveClassPercentAbove50);

        public RunSummary? LatestRun() =>
            image.Runs.Count == 0 ? null : image.Runs.MaxBy(r => r.DatePerformed);

        public bool IsAnalyzed() => image.Runs.Count > 0;
    }
}