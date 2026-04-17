using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Full manager for model packages: discover, import, load ONNX sessions, and delete.
/// Scans <c>ModelPackages/</c> for subdirectories containing a valid <c>manifest.json</c>,
/// exposes them as an observable list for UI binding, and loads ONNX sessions on demand.
/// Registered as a singleton in DI.
/// </summary>
public class ModelPackageService : IDisposable
{
    private static readonly string ModelPackagesDir =
        Path.Combine(AppContext.BaseDirectory, "ModelPackages");

    private ModelPackage? _loadedPackage;
    private string? _loadedPackageDir;

    /// <summary>Observable list of discovered model packages.</summary>
    public ObservableCollection<ModelManifest> AvailableModels { get; } = [];

    /// <summary>
    /// Scans the <c>ModelPackages/</c> directory and repopulates <see cref="AvailableModels"/>.
    /// Creates the directory if it doesn't exist.
    /// Safe to call multiple times (clears and re-scans).
    /// </summary>
    public void Refresh()
    {
        AvailableModels.Clear();

        // Ensure the directory exists (first launch or after deletion)
        if (!Directory.Exists(ModelPackagesDir))
        {
            Directory.CreateDirectory(ModelPackagesDir);
            Debug.WriteLine($"ModelPackageService: created {ModelPackagesDir}");
        }

        foreach (var dir in Directory.GetDirectories(ModelPackagesDir))
        {
            var result = TryLoadManifest(dir);
            if (result.Value != null)
                AvailableModels.Add(result.Value);
        }

        Debug.WriteLine($"ModelPackageService: found {AvailableModels.Count} model(s)");
    }


    /// <summary>
    /// Imports a model package from an external directory by copying it into
    /// <c>ModelPackages/</c>. Validates that the source contains a valid
    /// <c>manifest.json</c> and the referenced ONNX file before copying.
    /// </summary>
    public Result<ModelManifest> ImportPackage(string sourceDir)
    {
        if (!Directory.Exists(sourceDir))
            return Result<ModelManifest>.Fail($"Source directory not found: {sourceDir}");

        var manifestPath = Path.Combine(sourceDir, "manifest.json");
        if (!File.Exists(manifestPath))
            return Result<ModelManifest>.Fail("No manifest.json found in the selected folder.");

        // Parse manifest file
        ModelManifest manifest;
        try
        {
            var json = File.ReadAllText(manifestPath);
            manifest = JsonSerializer.Deserialize<ModelManifest>(json)
                       ?? throw new InvalidDataException("Failed to parse manifest.json");
        }
        catch (Exception ex)
        {
            return Result<ModelManifest>.Fail($"Invalid manifest.json: {ex.Message}");
        }

        var onnxFilename = manifest.Artifacts.ModelOnnx;
        if (string.IsNullOrWhiteSpace(onnxFilename))
            return Result<ModelManifest>.Fail("manifest.json is missing artifacts.model_onnx.");

        var onnxPath = Path.Combine(sourceDir, onnxFilename);
        if (!File.Exists(onnxPath))
            return Result<ModelManifest>.Fail($"ONNX file not found: {onnxFilename}");

        var folderName =
            Path.GetFileName(sourceDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        var targetDir = Path.Combine(ModelPackagesDir, folderName);

        if (Directory.Exists(targetDir))
            return Result<ModelManifest>.Fail($"A model package named '{folderName}' already exists.");

        try
        {
            Directory.CreateDirectory(ModelPackagesDir);
            CopyDirectory(sourceDir, targetDir);
        }
        catch (Exception ex)
        {
            return Result<ModelManifest>.Fail($"Failed to copy model package: {ex.Message}");
        }

        Refresh();
        var imported = AvailableModels.FirstOrDefault(m => m.Id == folderName);
        return imported != null
            ? Result<ModelManifest>.Ok(imported)
            : Result<ModelManifest>.Fail("Package copied but failed to load.");
    }


    /// <summary>
    /// Deletes a model package by removing its directory from <c>ModelPackages/</c>.
    /// </summary>
    public Result<bool> DeletePackage(string modelId)
    {
        var targetDir = Path.Combine(ModelPackagesDir, modelId);
        if (!Directory.Exists(targetDir))
            return Result<bool>.Fail($"Model package '{modelId}' not found.");

        // If the deleted package is currently loaded, dispose it
        if (_loadedPackageDir == Path.GetFullPath(targetDir))
        {
            _loadedPackage?.Dispose();
            _loadedPackage = null;
            _loadedPackageDir = null;
        }

        try
        {
            Directory.Delete(targetDir, recursive: true);
        }
        catch (Exception ex)
        {
            return Result<bool>.Fail($"Failed to delete model package: {ex.Message}");
        }

        Refresh();
        return Result<bool>.Ok(true);
    }

    /// <summary>
    /// Loads (or returns cached) the ONNX session for the given package directory.
    /// Caches the session, calling with the same path twice returns the same session.
    /// Calling with a different path disposes the previous session and loads the new one.
    /// </summary>
    public ModelPackage LoadPackage(string packageDir)
    {
        var fullPath = Path.GetFullPath(packageDir);

        // Return cached if same directory
        if (_loadedPackageDir == fullPath && _loadedPackage != null)
            return _loadedPackage;

        // Dispose previous session before loading new one
        _loadedPackage?.Dispose();

        var json = File.ReadAllText(Path.Combine(fullPath, "manifest.json"));
        var manifest = JsonSerializer.Deserialize<ModelManifest>(json)
                       ?? throw new InvalidDataException("Failed to parse manifest");

        var modelPath = Path.Combine(fullPath, manifest.Artifacts.ModelOnnx);
        var (session, provider) = CreateSession(modelPath);

        _loadedPackage = new ModelPackage
        {
            Manifest = manifest,
            Session = session,
            ActiveProvider = provider,
        };
        _loadedPackageDir = fullPath;

        return _loadedPackage;
    }

    /// <summary>
    /// Creates an ONNX InferenceSession with GPU fallback logic.
    /// Tries CUDA (Windows/Linux) or CoreML (macOS), falls back to CPU.
    /// </summary>
    private static (InferenceSession Session, ExecutionProvider Provider) CreateSession(string modelPath)
    {
        try
        {
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            ExecutionProvider provider;
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
            else
            {
                provider = ExecutionProvider.Cpu;
            }

            return (new InferenceSession(modelPath, options), provider);
        }
        catch
        {
            // GPU provider failed: fall back to CPU
            var fallback = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            return (new InferenceSession(modelPath, fallback), ExecutionProvider.Cpu);
        }
    }


    /// <summary>
    /// Attempts to parse a model package directory's manifest.
    /// Stamps <see cref="ModelManifest.Id"/> and <see cref="ModelManifest.DirectoryPath"/>
    /// for runtime use. Returns null if the directory is not a valid model package.
    /// Used both for scanning existing packages and previewing imports.
    /// </summary>
    public static Result<ModelManifest> TryLoadManifest(string packageDir)
    {
        var manifestPath = Path.Combine(packageDir, "manifest.json");
        if (!File.Exists(manifestPath))
            return Result<ModelManifest>.Fail("manifest.json not found");

        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ModelManifest>(json);
            if (manifest == null) 
                return Result<ModelManifest>.Fail("Failed to deserialize manifest");

            manifest.Id = Path.GetFileName(packageDir);
            manifest.DirectoryPath = Path.GetFullPath(packageDir);
            return Result<ModelManifest>.Ok(manifest);
        }
        catch (Exception ex)
        {
            return Result<ModelManifest>.Fail($"Manifest error: {ex.Message}");
        }
    }
    
    private static void CopyDirectory(string sourceDir, string targetDir)
    {
        Directory.CreateDirectory(targetDir);

        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var destFile = Path.Combine(targetDir, Path.GetFileName(file));
            File.Copy(file, destFile, overwrite: false);
        }

        foreach (var subDir in Directory.GetDirectories(sourceDir))
        {
            var destSubDir = Path.Combine(targetDir, Path.GetFileName(subDir));
            CopyDirectory(subDir, destSubDir);
        }
    }

    public void Dispose()
    {
        _loadedPackage?.Dispose();
        _loadedPackage = null;
        _loadedPackageDir = null;
        GC.SuppressFinalize(this);
    }
}