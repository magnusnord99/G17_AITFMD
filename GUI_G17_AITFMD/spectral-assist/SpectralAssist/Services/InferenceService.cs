using System;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Inference;

namespace SpectralAssist.Services;

/// <summary>
/// Runs ONNX model inference on a preprocessed HSI cube.
/// Single responsibility: set model → classify patches → return result.
/// Preprocessing is handled separately by <see cref="PreprocessingService"/>.
/// Model loading/caching is handled by <see cref="ModelPackageService"/>.
/// Registered as a singleton in DI.
/// </summary>
public class InferenceService
{
    private readonly Onnx3DCnnClassifier _onnx3DCnn = new();

    /// <summary>
    /// Runs ONNX inference on an already-preprocessed cube.
    /// </summary>
    /// <param name="preprocessedResult">Output from <see cref="PreprocessingService"/>.</param>
    /// <param name="package">Loaded model package (from <see cref="ModelPackageService"/>).</param>
    /// <param name="strideOverride">Optional stride override (null = use manifest default).</param>
    /// <param name="progress">Reports status messages back to the caller.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ClassificationResult> RunAsync(
        PreprocessingResult preprocessedResult,
        ModelPackage package,
        int? strideOverride = null,
        IProgress<string>? progress = null,
        CancellationToken ct = default)
    {
        _onnx3DCnn.SetModel(package);
        
        var patchProgress = progress != null
            ? new Progress<(int Done, int Total)>(p =>
                progress.Report($"ONNX inference... {p.Done}/{p.Total} patches ({100.0 * p.Done / p.Total:F0}%)"))
            : null;

        return await Task.Run(
            () => _onnx3DCnn.ClassifyImageAsync(preprocessedResult.Cube, preprocessedResult.TissueMask, strideOverride, patchProgress, ct), ct);
    }
}