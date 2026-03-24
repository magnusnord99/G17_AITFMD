using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public class PythonClassifier(string hdrPath) : IClassifier
{
    
    public async Task<ClassificationResult> ClassifyImageAsync(HsiCube cube)
    {
        var (success, _, json) = await Task.Run(RunPythonInference);

        if (!success)
            throw new InvalidOperationException($"Python inference failed: {json}");

        return ParsePredictionJson(json, cube);
    }
    
    
    private (bool Success, string? OutputDir, string Output) RunPythonInference()
    {
        var scriptPath = FindInferenceScript();
        if (!File.Exists(scriptPath))
            return (false, null, $"run_inference.py not found: {scriptPath}");

        var scriptDir = Path.GetDirectoryName(scriptPath) ?? ".";
        var outputDir = Path.Combine(Path.GetTempPath(), $"inference_{Guid.NewGuid():N}");
        Directory.CreateDirectory(outputDir);
        var predictionPath = Path.Combine(outputDir, "prediction.json");

        try
        {
            var pythonExe = FindPythonExe(scriptDir);
            var configArg = FindConfigArg(scriptDir);

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" --input \"{hdrPath}\" --output-dir \"{{outputDir}}\" {{configArg}}",
                WorkingDirectory = scriptDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            if (proc == null)
                return (false, null, "Failed to start Python process");

            var stdout = proc.StandardOutput.ReadToEnd();
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(TimeSpan.FromSeconds(120));

            if (proc.ExitCode != 0)
                return (false, null, $"Exit {proc.ExitCode}: {stderr}");

            var json = File.Exists(predictionPath)
                ? File.ReadAllText(predictionPath)
                : stdout;

            return (true, outputDir, json);
        }
        catch (Exception ex)
        {
            return (false, null, ex.Message);
        }
    }

    private static string FindInferenceScript()
    {
        var exeDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".";
        var dir = new DirectoryInfo(exeDir);
        for (var i = 0; i < 8 && dir != null; i++)
        {
            var candidate = Path.Combine(dir.FullName, "ML_PIPELINE_G17_AITFMD", "run_inference.py");
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }
        return Path.Combine(Directory.GetCurrentDirectory(),
            "..", "..", "..", "ML_PIPELINE_G17_AITFMD", "run_inference.py");
    }

    private static string FindPythonExe(string scriptDir)
    {
        // Try common venv locations
        var candidates = new[]
        {
            Path.Combine(scriptDir, "venv", "Scripts", "python.exe"),  // Windows
            Path.Combine(scriptDir, ".venv", "Scripts", "python.exe"), // Windows
            Path.Combine(scriptDir, "venv", "bin", "python"),          // Linux/macOS
            Path.Combine(scriptDir, ".venv", "bin", "python"),         // Linux/macOS
        };

        return candidates.FirstOrDefault(File.Exists) ?? "python";
    }

    private static string FindConfigArg(string scriptDir)
    {
        var configPath = Path.Combine(scriptDir, "configs", "inference", "pytorch.yaml");
        return File.Exists(configPath)
            ? $"--config \"{configPath}\""
            : "--config configs/inference/pytorch.yaml";
    }

    public static ClassificationResult ParsePredictionJson(string json, HsiCube cube)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // Validate status
        var status = root.GetProperty("status").GetString();
        if (status != "ok")
            throw new InvalidOperationException($"Python inference status: {status}");
        
        // Model info
        var modelInfo = root.GetProperty("model_info");
        var modelName = modelInfo.GetProperty("name").GetString() ?? "Python model";
        var classNames = modelInfo.GetProperty("class_names")
            .EnumerateArray()
            .Select(e => e.GetString()!)
            .ToList();

        // Spatial info
        var spatial = root.GetProperty("spatial");
        var patchH = spatial.GetProperty("patch_h").GetInt32();
        var patchW = spatial.GetProperty("patch_w").GetInt32();
        var cubeShape = spatial.GetProperty("cube_shape")
            .EnumerateArray()
            .Select(e => e.GetInt32())
            .ToArray();

        // Patch stats
        var stats = root.GetProperty("patch_stats");
        var totalPossible = stats.GetProperty("total_possible").GetInt32();
        var evaluated = stats.GetProperty("evaluated").GetInt32();
        var filtered = stats.GetProperty("filtered_by_tissue").GetInt32();

        // Parse each prediction
        var predictions = new List<PatchPrediction>();
        foreach (var pred in root.GetProperty("predictions").EnumerateArray())
        {
            var probabilities = new float[classNames.Count];
            var probsObj = pred.GetProperty("probabilities");
            for (var i = 0; i < classNames.Count; i++)
            {
                probabilities[i] = (float)probsObj.GetProperty(classNames[i]).GetDouble();
            }

            predictions.Add(new PatchPrediction
            {
                X = pred.GetProperty("x").GetInt32(),
                Y = pred.GetProperty("y").GetInt32(),
                Probabilities = probabilities,
            });
        }

        return new ClassificationResult
        {
            Predictions = predictions,
            ImageWidth = cube.Samples,
            ImageHeight = cube.Lines,
            PatchW = patchW,
            PatchH = patchH,
            Classes = classNames,
            ModelName = modelName,
            TotalPossible = totalPossible,
            Evaluated = evaluated,
            Skipped = filtered,
        };
    }
}
