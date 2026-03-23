using System;
using Microsoft.ML.OnnxRuntime;

namespace SpectralAssist.Models;

public class ModelPackage : IDisposable
{
    public required ModelManifest Manifest { get; init; }
    public required InferenceSession Session { get; init; }
    
    public void Dispose()
    {
        Session.Dispose();
        GC.SuppressFinalize(this);
    }
}