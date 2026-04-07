using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Hsi;

/// <summary>
/// Reflectance calibration: (scene - dark) / (white - dark + ε).
/// Normalizes raw sensor intensity to ~[0,1] surface reflectance.
/// </summary>
public static class HsiCalibration
{
    /// <summary>
    /// Looks for dark and white reference .hdr files in the same directory as the scene.
    /// Matches any .hdr file whose name contains "dark" or "white" (case-insensitive).
    /// Both must exist for calibration to proceed.
    /// </summary>
    private static bool TryFindReferenceHdrPaths(
        string sceneHdrPath,
        out string? darkHdrPath,
        out string? whiteHdrPath)
    {
        darkHdrPath = null;
        whiteHdrPath = null;
        var dir = Path.GetDirectoryName(sceneHdrPath);
        if (string.IsNullOrEmpty(dir))
            return false;

        foreach (var file in Directory.GetFiles(dir, "*.hdr"))
        {
            var name = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
            if (name.Contains("dark"))
                darkHdrPath = file;
            else if (name.Contains("white"))
                whiteHdrPath = file;
        }

        return darkHdrPath != null && whiteHdrPath != null;
    }

    /// <summary>
    /// Scans the scene's directory for "dark" and "white" reference .hdr files.
    /// If both are found, loads them and applies reflectance calibration.
    /// Returns null if either reference is missing (calibration is skipped).
    /// </summary>
    public static async Task<HsiCube?> TryCalibrateAsync(
        string sceneHdrPath, HsiCube sceneCube,
        IProgress<(string Status, double Progress)>? progress = null,
        CancellationToken ct = default)
    {
        if (!TryFindReferenceHdrPaths(sceneHdrPath, out var darkPath, out var whitePath))
            return null;

        progress?.Report(("Loading dark/white references...", 1));
        var darkTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(darkPath!), ct: ct);
        var whiteTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(whitePath!), ct: ct);
        await Task.WhenAll(darkTask, whiteTask);

        var dark = darkTask.Result;
        var white = whiteTask.Result;

        if (sceneCube.Bands != dark.Bands || sceneCube.Bands != white.Bands)
            throw new InvalidOperationException(
                $"Band mismatch: scene {sceneCube.Bands}, dark {dark.Bands}, white {white.Bands}");

        progress?.Report(("Calibration complete...", 1));
        return ApplyReflectance(sceneCube, dark, white);
    }

    /// <summary>
    /// Applies reflectance calibration: (scene - dark) / (white - dark).
    /// Supports both full-frame references (same dimensions as scene) and
    /// single-line references (one row, broadcast across all lines.
    /// </summary>
    public static HsiCube ApplyReflectance(HsiCube sceneCube, HsiCube darkCube, HsiCube whiteCube)
    {
        var header = sceneCube.Header;
        var bands = header.Bands;
        var samples = header.Samples;
        var lines = header.Lines;
        var pixels = lines * samples;
        var lineRef = darkCube.Header.Lines != lines;

        var result = new float[bands * pixels];

        // Each band is calibrated independently
        Parallel.For(0, bands, b =>
        {
            var sceneBand = sceneCube.GetBand(b);
            var darkBand = darkCube.GetBand(b);
            var whiteBand = whiteCube.GetBand(b);
            var offset = b * pixels;

            if (lineRef)
            {
                // Single-line reference: dark/white have one value per column (x).
                // Precompute 1 / (white-dark) per column to avoid repeated division.
                Span<float> invDenom = stackalloc float[samples];
                Span<float> darkCol = stackalloc float[samples];
                for (var x = 0; x < samples; x++)
                {
                    var d = whiteBand[x] - darkBand[x];
                    invDenom[x] = d > 1f ? 1f / d : 0f;
                    darkCol[x] = darkBand[x];
                }

                // Apply: reflectance = (scene - dark) * (1 / (white - dark))
                for (var y = 0; y < lines; y++)
                {
                    var rowStart = y * samples;
                    for (var x = 0; x < samples; x++)
                        result[offset + rowStart + x] =
                            (sceneBand[rowStart + x] - darkCol[x]) * invDenom[x];
                }
            }
            else
            {
                // Full-frame reference: one dark/white value per pixel
                // Apply: reflectance = (raw - dark) / (white - dark)
                for (var i = 0; i < pixels; i++)
                {
                    var d = whiteBand[i] - darkBand[i];
                    result[offset + i] = d > 1f
                        ? (sceneBand[i] - darkBand[i]) / d
                        : 0f;
                }
            }
        });

        return new HsiCube(header, result);
    }
}