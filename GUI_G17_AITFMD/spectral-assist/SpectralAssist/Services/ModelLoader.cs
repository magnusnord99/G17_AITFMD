using System;
using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public class ModelLoader : IDisposable
{
    /// <summary>Laster manifest + ONNX-session. Disponeres av <see cref="Onnx3DCnnClassifier.SetModel"/> / tilsvarende.</summary>
    public ModelPackage LoadPackage(string packageDir)
    {
        // Read manifest
        var json = File.ReadAllText(Path.Combine(packageDir, "manifest.json"));
        var manifest = JsonSerializer.Deserialize<ModelManifest>(json)
            ?? throw new InvalidDataException("Failed to parse manifest");

        // Load ONNX session
        var onnxPath = Path.Combine(packageDir, manifest.Artifacts.PipelineOnnx);
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        var session = new InferenceSession(onnxPath, options);

        return new ModelPackage
        {
            Manifest = manifest,
            Session = session
        };
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}
