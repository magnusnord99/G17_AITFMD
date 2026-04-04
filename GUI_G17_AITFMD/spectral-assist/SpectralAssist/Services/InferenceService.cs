using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Inference;

namespace SpectralAssist.Services;

/// <summary>
/// Owns the ONNX model lifecycle and orchestrates the inference pipeline:
/// resolve model → load/cache → preprocess → classify → format summary.
///
/// Registered as a singleton in DI so the model stays cached across image reloads.
/// </summary>
public class InferenceService : IDisposable
{
    private readonly ModelPackageLoader _packageLoader = new();
    private readonly Onnx3DCnnClassifier _onnx3DCnn = new();
    private string? _loadedModelPackageDir;

    /// <summary>
    /// Runs the full inference pipeline on a calibrated BSQ cube.
    /// </summary>
    /// <param name="calibratedCube">Already-calibrated cube from <see cref="ImageLoadingService"/>.</param>
    /// <param name="progress">Reports status messages back to the caller.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The classification result and a formatted summary string.</returns>
    public async Task<(ClassificationResult Result, string Summary)> RunAsync(
        HsiCube calibratedCube,
        IProgress<string>? progress = null,
        CancellationToken ct = default)
    {
        var packageDir = ResolveModelPackageDirectory();
        if (!Directory.Exists(packageDir))
        {
            throw new DirectoryNotFoundException(
                $"Model folder not found:\n{packageDir}\n\nPlace manifest.json + model.onnx under Assets/models/... and rebuild.");
        }

        // Load model (cached after first call)
        var classifier = EnsureOnnx3DClassifier(packageDir);
        var prepConfig = classifier.Manifest?.PreprocessingConfig ?? new PreprocessingConfig();

        // Run manifest-driven preprocessing on cached calibrated BSQ cube
        progress?.Report("Preprocessing (manifest-driven)...");
        var pipelineResult = await Task.Run(
            () => PreprocessingService.RunFromCalibrated(calibratedCube, prepConfig), ct);

        // ONNX inference with per-patch progress
        var patchProgress = progress != null
            ? new Progress<(int Done, int Total)>(p =>
                progress.Report($"ONNX inference... {p.Done}/{p.Total} patches ({100.0 * p.Done / p.Total:F0}%)"))
            : null;

        var result = await Task.Run(
            () => classifier.ClassifyImageAsync(pipelineResult.Cube, pipelineResult.TissueMask, patchProgress), ct);

        return (result, FormatResultSummary(result));
    }
    
    private static string ResolveModelPackageDirectory()
    {
        return Path.Combine(
            AppContext.BaseDirectory,
            "ModelPackages",
            "baseline_3dcnn_20260324_083658_last");
    }

    private Onnx3DCnnClassifier EnsureOnnx3DClassifier(string packageDir)
    {
        var fullPath = Path.GetFullPath(packageDir);
        if (_loadedModelPackageDir == fullPath && _onnx3DCnn.Manifest != null)
            return _onnx3DCnn;

        var package = _packageLoader.LoadPackage(fullPath);
        _onnx3DCnn.SetModel(package);
        _loadedModelPackageDir = fullPath;
        return _onnx3DCnn;
    }

    private static string FormatResultSummary(ClassificationResult result)
    {
        var text = new StringBuilder();
        text.AppendLine($"Model: {result.ModelName}");
        text.AppendLine($"Evaluated: {result.Evaluated} patches ({result.Skipped} skipped as background)");
        text.AppendLine();

        foreach (var pred in result.Predictions)
        {
            var className = result.Classes[pred.PredictedClass];
            text.AppendLine($"  ({pred.X},{pred.Y}): {className} ({pred.Confidence:P1})");
        }

        return text.ToString();
    }

    public void Dispose()
    {
        _onnx3DCnn.Dispose();
        _packageLoader.Dispose();
        GC.SuppressFinalize(this);
    }
}