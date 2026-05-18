using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Packaging;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.Services.Inference;

/// <summary>
/// Orchestrates inference: resolves the active model, computes stride from user
/// preferences and manifest, and dispatches to the classifier.
/// Registered as a singleton in DI.
/// </summary>
public class InferenceService(
    Onnx3DCnnClassifier classifier,
    SessionService settings,
    ModelPackageManager modelManager)
{
    /// <summary>
    /// Loads (or returns cached) package for the currently active model.
    /// Returns null if no model is selected.
    /// </summary>
    public ModelPackage? GetActivePackage()
    {
        var model = settings.ActiveModel;
        return model == null ? null : modelManager.LoadPackage(model.DirectoryPath);
    }
    
    /// <summary>
    /// Runs inference using the stride computed from user settings and the model manifest.
    /// </summary>
    /// <param name="preprocessed">Preprocessed input data.</param>
    /// <param name="package">Model package to run inference with.</param>
    /// <param name="patchProgress">Optional patch‑level progress reporter.</param>
    /// <param name="ct">Cancellation token.</param>
    public Task<ClassificationReport> RunAsync(
        PreprocessingResult preprocessed,
        ModelPackage package,
        IProgress<(int Done, int Total)>? patchProgress = null,
        CancellationToken ct = default)
        => RunAsync(preprocessed, package, ComputeStride(package.Manifest), patchProgress, ct);


    /// <summary>
    /// Runs inference with an explicit stride, overriding the user's configured stride.
    /// </summary>
    /// <param name="preprocessed">Preprocessed input data.</param>
    /// <param name="package">Model package to run inference with.</param>
    /// <param name="stride">Stride used when extracting patches.</param>
    /// <param name="patchProgress">Optional patch‑level progress reporter.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<ClassificationReport> RunAsync(
        PreprocessingResult preprocessed,
        ModelPackage package,
        int stride,
        IProgress<(int Done, int Total)>? patchProgress = null,
        CancellationToken ct = default)
    {
        classifier.SetModel(package);

        return await Task.Run(
            () => classifier.ClassifyImageAsync(
                preprocessed.Cube, preprocessed.TissueMask, stride, patchProgress, ct),
            ct);
    }

    private int ComputeStride(ModelManifest manifest)
    {
        var patchSize = manifest.InputSpec.SpatialPatchSize[0];
        return settings.SelectedStride.Divisor switch
        {
            0  => manifest.InputSpec.Stride.FirstOrDefault(patchSize),
            var d => patchSize / d,
        };
    }
}