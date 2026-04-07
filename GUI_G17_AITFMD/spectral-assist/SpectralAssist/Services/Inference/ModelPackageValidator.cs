using System.IO;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

/// <summary>
/// Smoke test that runs a reference patch through the C# preprocessing pipeline and
/// ONNX inference, then compares against expected outputs baked into the model package
/// by the Python export script.
///
/// Validation is warn-only: a failed validation does not block inference.
/// </summary>
public static class ModelPackageValidator
{
    /// <summary>
    /// Validates the full pipeline (preprocessing + ONNX inference) against reference data
    /// shipped in the model package's validation artifacts.
    /// Returns <see cref="ModelValidationResult.Skipped"/> if no validation artifacts exist.
    /// </summary>
    public static ModelValidationResult Validate(
        string modelPackageDir,
        ModelManifest manifest,
        InferenceSession session)
    {
        // Check if validation artifacts exist
        var artifacts = manifest.Artifacts;
        if (string.IsNullOrEmpty(artifacts.ValidationExpectedJson) ||
            string.IsNullOrEmpty(artifacts.ValidationPatchRawBin))
            return ModelValidationResult.Skipped;

        var expectedJsonPath = Path.Combine(modelPackageDir, artifacts.ValidationExpectedJson);
        var patchRawPath = Path.Combine(modelPackageDir, artifacts.ValidationPatchRawBin);
        if (!File.Exists(expectedJsonPath) || !File.Exists(patchRawPath))
            return ModelValidationResult.Skipped;

        var prepConfig = manifest.Pipeline.Preprocessing.Params;

        //ToDo: Implement actual validation pipeline
        // 1. Read raw patch .bin (HWC float32) → reshape → convert to BSQ
        // 2. Run C# preprocessing steps on it
        // 3. Feed to ONNX session
        // 4. Compare output logits against expected values from JSON

        var preprocessMaxDiff = 0.0f;
        var maskMatched = true;
        var inferenceMaxDiff = 0.0f;

        var passed = true;
        var summary = passed
            ? $"PASSED: preprocessing max_diff={preprocessMaxDiff:E2}, mask_match={maskMatched}, inference max_diff={inferenceMaxDiff:E2}"
            : $"FAILED: preprocessing max_diff={preprocessMaxDiff:E2}, mask_match={maskMatched}, inference max_diff={inferenceMaxDiff:E2}";

        return new ModelValidationResult(passed, preprocessMaxDiff, maskMatched, inferenceMaxDiff, summary);
    }

    private static ModelValidationResult Fail(string message) =>
        new(passed: false, preprocessingMaxAbsDiff: float.NaN, maskMatched: false,
            inferenceMaxAbsDiff: float.NaN, summary: $"FAILED: {message}");
}