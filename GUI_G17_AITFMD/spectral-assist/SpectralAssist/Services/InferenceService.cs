using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Inference;

namespace SpectralAssist.Services;

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
    /// Loads (or returns cached) the package for the currently active model.
    /// Returns null if no model is selected.
    /// </summary>
    public ModelPackage? GetActivePackage()
    {
        var model = settings.ActiveModel;
        return model == null ? null : modelManager.LoadPackage(model.DirectoryPath);
    }
    
    /// <summary>
    /// Runs inference using the stride derived from user settings and the manifest.
    /// </summary>
    public Task<ClassificationReport> RunAsync(
        PreprocessingResult preprocessed,
        ModelPackage package,
        IProgress<string>? progress = null,
        CancellationToken ct = default)
        => RunAsync(preprocessed, package, ComputeStride(package.Manifest), progress, ct);
    
    
    /// <summary>
    /// Runs inference with an explicit stride. Use when callers want to override the user's current stride preference.
    /// </summary>
    public async Task<ClassificationReport> RunAsync(
        PreprocessingResult preprocessed,
        ModelPackage package,
        int stride,
        IProgress<string>? progress,
        CancellationToken ct)
    {
        classifier.SetModel(package);

        var patchProgress = progress != null
            ? new Progress<(int Done, int Total)>(p =>
                progress.Report($"ONNX inference... {p.Done}/{p.Total} patches ({100.0 * p.Done / p.Total:F0}%)"))
            : null;

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