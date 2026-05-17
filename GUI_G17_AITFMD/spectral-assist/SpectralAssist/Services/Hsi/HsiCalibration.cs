using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Hsi;

/// <summary>
/// Reflectance calibration: (scene - dark) / (white - dark + ε).
/// Normalizes raw sensor intensity to between 0 and 1 surface reflectance.
/// </summary>
public static class HsiCalibration
{
    /// <summary>
    /// Returns true when both dark and white reference .hdr files exist in
    /// the same directory as the scene. Use to decide ahead of time whether
    /// the load pipeline will perform calibration.
    /// </summary>
    public static bool HasReferenceFiles(string sceneHdrPath)
        => TryFindReferenceHdrPaths(sceneHdrPath, out _, out _);

    /// <summary>
    /// Loads dark and white reference cubes from the scene's directory.
    /// Returns null if either reference is missing.
    /// </summary>
    public static async Task<(HsiCube Dark, HsiCube White)?> LoadReferencesAsync(
        string sceneHdrPath, CancellationToken ct = default)
    {
        if (!TryFindReferenceHdrPaths(sceneHdrPath, out var darkPath, out var whitePath))
            return null;

        var darkTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(darkPath!), ct: ct);
        var whiteTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(whitePath!), ct: ct);
        await Task.WhenAll(darkTask, whiteTask);

        return (darkTask.Result, whiteTask.Result);
    }

    /// <summary>
    /// Looks for dark and white reference .hdr files in the same directory as the scene.
    /// Matches any .hdr file whose name contains "dark" or "white" (case-insensitive).
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
    /// Applies reflectance calibration: <c>(scene - dark) / (white - dark + ε)</c>.
    /// Matches the Python pipeline (epsilon-based denominator).
    /// Supports both full-frame and single-line references.
    /// Reports band-level progress (fraction of bands completed) so callers can
    /// drive a determinate progress bar during the heavy parallel work.
    /// </summary>
    public static HsiCube ApplyReflectance(
        HsiCube sceneCube, HsiCube darkCube, HsiCube whiteCube,
        float eps = 1e-8f,
        IProgress<float>? bandProgress = null,
        CancellationToken ct = default)
    {
        var header = sceneCube.Header;
        var bands = header.Bands;
        var samples = header.Samples;
        var lines = header.Lines;
        var pixels = lines * samples;
        var lineRef = darkCube.Header.Lines != lines;

        var result = new float[bands * pixels];
        var completed = 0;

        // Each band is calibrated independently
        Parallel.For(0, bands, new ParallelOptions { CancellationToken = ct }, b =>
        {
            var sceneBand = sceneCube.GetBand(b);
            var darkBand = darkCube.GetBand(b);
            var whiteBand = whiteCube.GetBand(b);
            var offset = b * pixels;

            if (lineRef)
            {
                // Single-line reference: dark/white have one value per column (x).
                // Precompute 1 / (white - dark + eps) per column to avoid repeated division.
                Span<float> invDenom = stackalloc float[samples];
                Span<float> darkCol = stackalloc float[samples];
                for (var x = 0; x < samples; x++)
                {
                    invDenom[x] = 1f / (whiteBand[x] - darkBand[x] + eps);
                    darkCol[x] = darkBand[x];
                }

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
                for (var i = 0; i < pixels; i++)
                {
                    var denom = whiteBand[i] - darkBand[i] + eps;
                    result[offset + i] = (sceneBand[i] - darkBand[i]) / denom;
                }
            }

            var done = Interlocked.Increment(ref completed);
            bandProgress?.Report((float)done / bands);
        });

        return new HsiCube(header, result);
    }
}
