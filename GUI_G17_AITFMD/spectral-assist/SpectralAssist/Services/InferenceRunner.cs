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

    public static (bool Success, string Output) Run(string hdrPath)
    {
        var scriptPath = GetRunInferencePath();
        if (!File.Exists(scriptPath))
            return (false, $"run_inference.py not found: {scriptPath}");

        var outputPath = Path.Combine(Path.GetTempPath(), $"inference_{Guid.NewGuid():N}.json");
        try
        {
            // Use venv Python if available, otherwise fall back to system Python
            var scriptDir = Path.GetDirectoryName(scriptPath) ?? ".";
            var venvPython = Path.Combine(scriptDir, "venv", "Scripts", "python.exe");
            var pythonExe = File.Exists(venvPython) ? venvPython : "python";
 
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" --input \"{hdrPath}\" --output \"{outputPath}\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var proc = Process.Start(psi);
            if (proc == null)
                return (false, "Failed to start process");

            var stdout = proc.StandardOutput.ReadToEnd();
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(TimeSpan.FromSeconds(30));

            if (proc.ExitCode != 0)
                return (false, $"Exit {proc.ExitCode}: {stderr}");

            var json = File.Exists(outputPath) ? File.ReadAllText(outputPath) : stdout;
            return (true, json);
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
        finally
        {
            if (File.Exists(outputPath))
                try { File.Delete(outputPath); } catch { /* ignore */ }
        }
    }
}
