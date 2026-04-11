using System;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

public interface IClassifier
{
    Task<ClassificationResult> ClassifyImageAsync(
        HsiCube cube,
        bool[]? tissueMask = null,
        int? strideOverride = null,
        IProgress<(int Done, int Total)>? progress = null,
        CancellationToken ct = default
    );
}