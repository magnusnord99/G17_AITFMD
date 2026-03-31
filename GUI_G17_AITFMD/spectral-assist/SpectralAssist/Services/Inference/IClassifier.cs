using System;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

public interface IClassifier
{
    Task<ClassificationResult> ClassifyImageAsync(
        HsiCube cube,
        bool[]? tissueMask = null,
        IProgress<(int Done, int Total)>? progress = null);
}