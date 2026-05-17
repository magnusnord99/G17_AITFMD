using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Full manager for model packages: discover, import, load ONNX sessions, and delete.
/// Scans <c>ModelPackages/</c> for subdirectories containing a valid <c>manifest.json</c>,
/// exposes them as an observable list for UI binding, and loads ONNX sessions on demand.
/// </summary>
public class ModelPackageManager : IDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };
    private static readonly string ShippedModelsDir = Path.Combine(AppContext.BaseDirectory, "ModelPackages");
    private static readonly string UserModelsDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "SpectralAssist",
            "ModelPackages");
    
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
        
        if (!Directory.Exists(UserModelsDir)) 
            Directory.CreateDirectory(UserModelsDir);
        
        // User-imported models (writable)
        foreach (var dir in Directory.GetDirectories(UserModelsDir))
        { 
            var result = TryLoadManifest(dir); 
            if (result.Value != null) 
                AvailableModels.Add(result.Value);
        }
        
        // Shipped default models (read-only)
        if (Directory.Exists(ShippedModelsDir))
        {
            foreach (var dir in Directory.GetDirectories(ShippedModelsDir)) 
            { 
                var result = TryLoadManifest(dir); 
                if (result.Value != null) AvailableModels.Add(result.Value); 
            }
            
        }
        
        Debug.WriteLine($"ModelPackageService: found {AvailableModels.Count} model(s)");
    }


    /// <summary>
    /// Imports a model package from an external directory by copying it into
    /// <c>ModelPackages/</c>. Validates that the source contains a valid
    /// <c>manifest.json</c> and the referenced ONNX file before copying.
    /// </summary>
    public ServiceResult<ModelManifest> ImportPackage(string sourceDir)
    {
        if (!Directory.Exists(sourceDir))
            return ServiceResult<ModelManifest>.Fail($"Source directory not found: {sourceDir}");

        var preview = TryLoadManifest(sourceDir);
        if (!preview.IsSuccess || preview.Value == null)
            return ServiceResult<ModelManifest>.Fail(preview.Error ?? "Invalid model package");

        var sourceManifest = preview.Value;

        // Check if model is already imported through ID hashes
        var existing = AvailableModels.FirstOrDefault(m => m.Metadata.Id == sourceManifest.Metadata.Id);
        if (existing != null)
            return ServiceResult<ModelManifest>.Fail(
                $"This model is already imported as '{existing.DisplayName}'.");

        var folderName =
            Path.GetFileName(sourceDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        var targetDir = Path.Combine(UserModelsDir, folderName);

        if (Directory.Exists(targetDir))
            return ServiceResult<ModelManifest>.Fail(
                $"Folder '{folderName}' already exists. Remove it manually if it's a stale import.");

        // Copy model package files to app directory
        try
        {
            Directory.CreateDirectory(UserModelsDir);
            CopyDirectory(sourceDir, targetDir);
        }
        catch (Exception ex)
        {
            try
            {
                if (Directory.Exists(targetDir))
                    Directory.Delete(targetDir, recursive: true);
            }
            catch
            {
                // Clean up failed; continue
            }

            return ServiceResult<ModelManifest>.Fail($"Failed to copy model package: {ex.Message}");
        }

        Refresh();
        var imported = AvailableModels.FirstOrDefault(m => m.Metadata.Id == sourceManifest.Metadata.Id);
        return imported != null
            ? ServiceResult<ModelManifest>.Ok(imported)
            : ServiceResult<ModelManifest>.Fail("Package copied but failed to load.");
    }


    /// <summary>
    /// Deletes a model package by removing its directory from <c>ModelPackages/</c>.
    /// </summary>
    public ServiceResult<bool> DeletePackage(string modelId)
    {
        var manifest = AvailableModels.FirstOrDefault(m => m.Metadata.Id == modelId);
        if (manifest == null)
            return ServiceResult<bool>.Fail($"Model package '{modelId}' not found.");

        var targetDir = manifest.DirectoryPath;
        
        // Prevent deletion of shipped default models
        if (targetDir.StartsWith(ShippedModelsDir, StringComparison.OrdinalIgnoreCase)) 
            return ServiceResult<bool>.Fail("Cannot delete a default model that ships with the application.");

        // If the deleted package is currently loaded, dispose it
        if (_loadedPackageDir == targetDir)
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
            return ServiceResult<bool>.Fail($"Failed to delete model package: {ex.Message}");
        }

        Refresh();
        return ServiceResult<bool>.Ok(true);
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
    public static ServiceResult<ModelManifest> TryLoadManifest(string packageDir)
    {
        var manifestPath = Path.Combine(packageDir, "manifest.json");
        if (!File.Exists(manifestPath))
            return ServiceResult<ModelManifest>.Fail("manifest.json not found");

        try
        {
            var json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ModelManifest>(json);
            if (manifest == null)
                return ServiceResult<ModelManifest>.Fail("Failed to deserialize manifest");

            //manifest.Id = Path.GetFileName(packageDir);
            manifest.DirectoryPath = Path.GetFullPath(packageDir);


            if (string.IsNullOrEmpty(manifest.Metadata.Id))
            {
                var onnxPath = Path.Combine(packageDir, manifest.Artifacts.ModelOnnx);
                if (!File.Exists(onnxPath))
                    return ServiceResult<ModelManifest>.Fail("Failed to generate unique id (model.onnx missing");

                manifest.Metadata.Id = ComputeModelHash(onnxPath);

                try
                {
                    var updated = JsonSerializer.Serialize(manifest, JsonOpts);
                    File.WriteAllText(manifestPath, updated);
                }
                catch (Exception e)
                {
                    // ModelManifest.json ID update write failed; skip for now
                }
            }

            return ServiceResult<ModelManifest>.Ok(manifest);
        }
        catch (Exception ex)
        {
            return ServiceResult<ModelManifest>.Fail($"Manifest error: {ex.Message}");
        }
    }

    private static void CopyDirectory(string sourceDir, string targetDir)
    {
        Directory.CreateDirectory(targetDir);

        foreach (var file in Directory.GetFiles(sourceDir))
        {
            var fileName = Path.GetFileName(file);
            // Skip macOS AppleDouble metadata files (._*) and other OS artifacts
            if (fileName.StartsWith("._") || fileName == ".DS_Store")
                continue;

            var destFile = Path.Combine(targetDir, fileName);
            File.Copy(file, destFile, overwrite: false);
        }

        foreach (var subDir in Directory.GetDirectories(sourceDir))
        {
            var fileName = Path.GetFileName(subDir);
            if (fileName.StartsWith('.'))
                continue;

            var destSubDir = Path.Combine(targetDir, fileName);
            CopyDirectory(subDir, destSubDir);
        }
    }

    /// <summary>
    /// Computes a short SHA-256 prefix over the .onnx weights to use as a unique ID across imports.
    /// Initially tries to hash the onnx.data if it exists, falls back to onnx file if not.
    /// </summary>
    private static string ComputeModelHash(string onnxPath)
    {
        var dataPath = onnxPath + ".data";
        var pathToHash = File.Exists(dataPath) ? dataPath : onnxPath;

        using var stream = File.OpenRead(pathToHash);
        var hash = SHA256.HashData(stream);

        return "sha256-" + Convert.ToHexString(hash)[..16].ToLowerInvariant();
    }

    public void Dispose()
    {
        _loadedPackage?.Dispose();
        _loadedPackage = null;
        _loadedPackageDir = null;
        GC.SuppressFinalize(this);
    }
}