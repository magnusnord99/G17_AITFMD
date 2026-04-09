using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// The model package import smoke test validation.
/// Feeds a reference patch through preprocessing + ONNX inference and 
/// compares against expected outputs from the Python export script.
/// Validation is warn-only: a failed validation does not block inference.
/// </summary>
public static class ModelPackageValidator
{
    public static async Task<(bool Passed, string Summary)> ValidateAsync(
        string packageDir, ModelManifest manifest, InferenceService inferenceService)
    {
        var validationInfo = manifest.Validation;
        
        if (string.IsNullOrEmpty(validationInfo.RoiDir) ||
            validationInfo.ExpectedOutput?.Logits is not { Count: > 0 })
            return (false, "Skipped: missing validation data in package.");

        var validationDir = Path.Combine(packageDir, validationInfo.RoiDir);
        if (!Directory.Exists(validationDir))
            return (false, "Skipped: validation directory not found.");

        var preprocessingInfo = manifest.Pipeline.Preprocessing;
        
        try
        {
            // Step 1: Parse .hdr, load image, perform calibration:
            var imageResult = await ImageLoadingService.LoadAsync(Path.Combine(validationDir, "raw.hdr"));
            if (preprocessingInfo.Steps.Contains("calibrate") && !imageResult.HasCalibration)
                return (false, "Skipped: calibration white/black files missing");
            
            // Step 2: Perform remaining preprocessing steps
            //var preprocessingResult = PreprocessingService.RunFromCalibrated(imageResult.CalibratedCube, preprocessingInfo);
            
            // Step 3: Perform model inference
            var (classificationResult, _) = await inferenceService.RunAsync(imageResult.Cube, packageDir);

            // Step 4: Compare actual against expected
            var actualSoftmax = classificationResult.Predictions[0].Probabilities;
            var actualLogits = classificationResult.Predictions[0].Logits;
            var predictedClass = classificationResult.Predictions[0].PredictedClass;
            
            var expectedLogits = validationInfo.ExpectedOutput.Logits;
            var maxLogitDiff = MaxAbsDiff(actualLogits, expectedLogits);
            var absoluteTolerance = validationInfo.Tolerance;
            var passed = maxLogitDiff < absoluteTolerance;
            
            var actual = new ActualOutput
            {
                Logits = actualLogits.ToList(),
                Softmax = actualSoftmax.ToList(),
                PredictedClass = predictedClass,
                MaxLogitDiff = maxLogitDiff,
                ValidatedAt = DateTime.UtcNow.ToString(CultureInfo.InvariantCulture),
            };
            
            var summary = passed
                ? $"PASSED: max logit difference {maxLogitDiff:E2} within tolerance {absoluteTolerance:E2}"
                : $"FAILED: max logit difference {maxLogitDiff:E2} exceeds tolerance {absoluteTolerance:E2}";
            
            WriteResult(packageDir, passed, summary, actual);
            validationInfo.Status = passed ? "passed" : "failed";
            validationInfo.Summary = summary;
            validationInfo.ActualOutput = actual;
            
            return (passed, summary);
        }
        catch (Exception ex)
        {
            WriteResult(packageDir, passed: false, $"ERROR: {ex.Message}", actualOutput: null);
            return (false, $"Validation error: {ex.Message}");
        }
    }
    
    private static float MaxAbsDiff(float[] actual, List<float> expected)
    {
        var max = 0f;
        var len = Math.Min(actual.Length, expected.Count);
        for (var i = 0; i < len; i++)
        {
            var d = MathF.Abs(actual[i] - expected[i]);
            if (d > max) max = d;
        }
        return max;
    }
    
    private static void WriteResult(string packageDir, bool passed, string summary, ActualOutput? actualOutput)
    {
        var path = Path.Combine(packageDir, "manifest.json");
        var node = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))!;
        node["validation"]!["status"] = passed ? "passed" : "failed";
        node["validation"]!["summary"] = summary;

        if (actualOutput != null)
        {
            node["validation"]!["actual_output"] = JsonSerializer.SerializeToNode(actualOutput);
        }

        File.WriteAllText(path, node.ToJsonString(
            new JsonSerializerOptions { WriteIndented = true }));
    }
}