using System;
using Microsoft.ML.OnnxRuntime;

namespace SpectralAssist.Models;

public class ModelPackage : IDisposable
{
    public required ModelManifest Manifest { get; init; }
    public required InferenceSession Session { get; init; }
    public ExecutionProvider ActiveProvider { get; init; } = ExecutionProvider.Cpu;
    
    public void Dispose()
    {
        Session.Dispose();
        GC.SuppressFinalize(this);
    }
}

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