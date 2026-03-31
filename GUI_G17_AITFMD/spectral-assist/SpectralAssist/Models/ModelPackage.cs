using System;
using Microsoft.ML.OnnxRuntime;

namespace SpectralAssist.Models;

/// <summary>Which ONNX Runtime execution provider is active.</summary>
public enum ExecutionProvider
{
    /// <summary>CPU only: works on all platforms.</summary>
    Cpu,

    /// <summary>NVIDIA CUDA: Requires CUDA 12.x + cuDNN 9.x installed.</summary>
    Cuda,

    /// <summary>DirectML via DirectX 12: Any Windows GPU (NVIDIA/AMD/Intel), zero setup required.</summary>
    DirectML,
    
    /// <summary>CoreML: Apple's performant machine learning framework.</summary>
    CoreML,
}

public class ModelPackage : IDisposable
{
    public required ModelManifest Manifest { get; init; }
    public required InferenceSession Session { get; init; }

    /// <summary>Whether this session was created with a GPU execution provider.</summary>
    public bool UseGpu { get; init; }

    /// <summary>Which execution provider is actually active for this session.</summary>
    public ExecutionProvider ActiveProvider { get; init; } = ExecutionProvider.Cpu;

    /// <summary>Result of the smoke-test validation run at load time. Null if no validation artifacts were present.</summary>
    public ModelValidationResult? ValidationResult { get; set; }

    public void Dispose()
    {
        Session.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Outcome of the model-package smoke test (preprocessing parity + ONNX inference parity).</summary>
public readonly struct ModelValidationResult(
    bool passed,
    float preprocessingMaxAbsDiff,
    bool maskMatched,
    float inferenceMaxAbsDiff,
    string summary)
{
    public bool Passed { get; } = passed;
    public float PreprocessingMaxAbsDiff { get; } = preprocessingMaxAbsDiff;
    public bool MaskMatched { get; } = maskMatched;
    public float InferenceMaxAbsDiff { get; } = inferenceMaxAbsDiff;
    public string Summary { get; } = summary;

    /// <summary>Returned when the model package has no validation/folder (backwards compatible).</summary>
    public static ModelValidationResult Skipped { get; } = new(
        passed: true,
        preprocessingMaxAbsDiff: 0f,
        maskMatched: true,
        inferenceMaxAbsDiff: 0f,
        summary: "Validation skipped: no validation artifacts in model package.");
}