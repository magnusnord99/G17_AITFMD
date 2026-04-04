using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

public class ModelPackageLoader : IDisposable
{
    /// <summary>
    /// Loads manifest + ONNX session. When <paramref name="useGpu"/> is true, tries CUDA and DirectML.
    /// Falls back to DirectML if CUDA/cuDNN is not installed.
    /// If GPU accelerated providers fails, silently falls back to CPU.
    /// </summary>
    public ModelPackage LoadPackage(string packageDir, bool useGpu = false)
    {
        // Read manifest
        var json = File.ReadAllText(Path.Combine(packageDir, "manifest.json"));
        var manifest = JsonSerializer.Deserialize<ModelManifest>(json)
                       ?? throw new InvalidDataException("Failed to parse manifest");

        // Load ONNX session
        var modelPath = Path.Combine(packageDir, manifest.Artifacts.PipelineOnnx);
        var provider = ExecutionProvider.Cpu;
        InferenceSession? session;
        
        try
        {
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                options.AppendExecutionProvider_CUDA(deviceId: 0);
                provider = ExecutionProvider.Cuda;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                options.AppendExecutionProvider_CoreML();
                provider = ExecutionProvider.CoreML;
            }

            session = new InferenceSession(modelPath, options);
        }
        catch
        {
            // GPU provider failed: Fall back to CPU with clean options
            provider = ExecutionProvider.Cpu;
            var fallbackOptions = new SessionOptions();
            fallbackOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            session = new InferenceSession(modelPath, fallbackOptions);
        }
        
        // ToDo: Run smoke test or some kind of validation if artifacts are present /validation folder?
        //var validationResult = ModelPackageValidator.Validate(packageDir, manifest, session); example
        var validationResult = ModelValidationResult.Skipped;

        return new ModelPackage
        {
            Manifest = manifest,
            Session = session,
            UseGpu = provider != ExecutionProvider.Cpu,
            ActiveProvider = provider,
            ValidationResult = validationResult,
        };
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}