using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Inference;

namespace SpectralAssist.Services;

/// <summary>
/// Orchestrates the inference pipeline: load model → preprocess → classify → format summary.
/// Delegates model loading/caching to <see cref="ModelPackageService"/>.
/// Registered as a singleton in DI.
/// </summary>
public class InferenceService(ModelPackageService packageService)
{
    private readonly Onnx3DCnnClassifier _onnx3DCnn = new();

    /// <summary>
    /// Runs the full inference pipeline on a calibrated BSQ cube.
    /// </summary>
    /// <param name="calibratedCube">Already-calibrated cube from <see cref="ImageLoadingService"/>.</param>
    /// <param name="modelPackageDir">Path to the model package directory to use.</param>
    /// <param name="progress">Reports status messages back to the caller.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task<(ClassificationResult Result, string Summary)> RunAsync(
        HsiCube calibratedCube,
        string modelPackageDir,
        IProgress<string>? progress = null,
        CancellationToken ct = default)
    {
        if (!Directory.Exists(modelPackageDir))
        {
            throw new DirectoryNotFoundException(
                $"Model folder not found:\n{modelPackageDir}\n\nImport a model package via the Models page.");
        }

        // Load model (cached by ModelPackageService if same directory)
        var package = packageService.LoadPackage(modelPackageDir);
        _onnx3DCnn.SetModel(package);

        // Run manifest-driven preprocessing on cached calibrated BSQ cube
        progress?.Report("Preprocessing (manifest-driven)...");
        var preprocessing = package.Manifest.Pipeline.Preprocessing;
        var pipelineResult = await Task.Run(
            () => PreprocessingService.RunFromCalibrated(calibratedCube, preprocessing), ct);

        // ONNX inference with per-patch progress
        var patchProgress = progress != null
            ? new Progress<(int Done, int Total)>(p =>
                progress.Report($"ONNX inference... {p.Done}/{p.Total} patches ({100.0 * p.Done / p.Total:F0}%)"))
            : null;

        var result = await Task.Run(
            () => _onnx3DCnn.ClassifyImageAsync(pipelineResult.Cube, pipelineResult.TissueMask, patchProgress), ct);

        return (result, FormatResultSummary(result));
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
}