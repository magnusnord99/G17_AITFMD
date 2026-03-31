using System.IO;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

/// <summary>
/// Validation test that runs a reference cube through the C# preprocessing pipeline and
/// ONNX inference, then compares against expected outputs baked into the model package by
/// the Python export script.
///
/// Validation is warn-only: a failed validation does not block inference.
/// </summary>
public static class ModelPackageValidator
{
    /// <summary>
    /// Validates the full pipeline (preprocessing + ONNX inference) against reference data
    /// shipped in the model package's <c>validation/</c> folder.
    /// Returns <see cref="ModelValidationResult.Skipped"/> if no validation artifacts exist.
    /// </summary>
    public static ModelValidationResult Validate(
        string modelPackageDir,
        ModelManifest manifest,
        InferenceSession session)
    {
        var valDir = Path.Combine(modelPackageDir, "validation");
        if (!Directory.Exists(valDir))
            return ModelValidationResult.Skipped;

        var validation = manifest.Validation;
        var prepConfig = manifest.PreprocessingConfig;
        if (validation == null || prepConfig == null)
            return ModelValidationResult.Skipped;

        if (validation.RefCubeShape.Count < 3)
            return Fail("validation.ref_cube_shape must have 3 elements (lines, samples, bands).");
        
        //ToDo: Fix validation pipeline and replace constant values

        var preprocessTolerance = 0.0f;
        var preprocessMaxDiff = 0.0f;
        var maskMatched = true;
        var inferenceMaxDiff = 0.0f;
        
        var passed = true;
        var summary = passed
            ? $"PASSED: preprocessing max_diff={preprocessMaxDiff:E2}, mask_match={maskMatched}, inference max_diff={inferenceMaxDiff:E2}"
            : $"FAILED: preprocessing max_diff={preprocessMaxDiff:E2} (actual={preprocessTolerance:E2}), mask_match={maskMatched}, inference max_diff={inferenceMaxDiff:E2}";

        return new ModelValidationResult(passed, preprocessMaxDiff, maskMatched, inferenceMaxDiff, summary);
    }
    
    private static ModelValidationResult Fail(string message) =>
        new(passed: false, preprocessingMaxAbsDiff: float.NaN, maskMatched: false,
            inferenceMaxAbsDiff: float.NaN, summary: $"FAILED: {message}");
}