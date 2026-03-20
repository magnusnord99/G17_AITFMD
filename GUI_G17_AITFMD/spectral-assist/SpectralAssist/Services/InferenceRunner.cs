using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace SpectralAssist.Services;

public static class InferenceRunner
{
    private static string GetRunInferencePath()
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
        return Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "ML_PIPELINE_G17_AITFMD", "run_inference.py");
    }

    /// <summary>
    /// Runs inference and returns (Success, OutputDir, Json). Se prediction.json for patches/spatial (heatmap bygges i GUI om ønskelig).
    /// </summary>
    public static (bool Success, string? OutputDir, string Output) Run(string hdrPath)
    {
        var scriptPath = GetRunInferencePath();
        if (!File.Exists(scriptPath))
            return (false, null, $"run_inference.py not found: {scriptPath}");

        var outputDir = Path.Combine(Path.GetTempPath(), $"inference_{Guid.NewGuid():N}");
        Directory.CreateDirectory(outputDir);
        var predictionPath = Path.Combine(outputDir, "prediction.json");

        try
        {
            var scriptDir = Path.GetDirectoryName(scriptPath) ?? ".";
            var venvPython = Path.Combine(scriptDir, ".venv", "bin", "python");
            if (!File.Exists(venvPython))
                venvPython = Path.Combine(scriptDir, "venv", "Scripts", "python.exe");
            var pythonExe = File.Exists(venvPython) ? venvPython : "python";

            var configPath = Path.Combine(scriptDir, "configs", "inference", "pytorch.yaml");
            var configArg = File.Exists(configPath)
                ? $"--config \"{configPath}\""
                : "--config configs/inference/pytorch.yaml";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" --input \"{hdrPath}\" --output-dir \"{outputDir}\" {configArg}",
                WorkingDirectory = scriptDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var proc = Process.Start(psi);
            if (proc == null)
                return (false, null, "Failed to start process");

            var stdout = proc.StandardOutput.ReadToEnd();
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(TimeSpan.FromSeconds(120));

            if (proc.ExitCode != 0)
                return (false, null, $"Exit {proc.ExitCode}: {stderr}");

            var json = File.Exists(predictionPath) ? File.ReadAllText(predictionPath) : stdout;
            return (true, outputDir, json);
        }
        catch (Exception ex)
        {
            return (false, null, ex.Message);
        }
    }
}
