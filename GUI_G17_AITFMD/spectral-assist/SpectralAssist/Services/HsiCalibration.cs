using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Converts raw HSI intensity values into reflectance using dark and white reference captures.
///
/// The standard reflectance formula is:
///   reflectance = (raw - dark) / (white - dark)
///
/// Where:
///   raw   = the scene measurement (signal + noise + dark current)
///   dark  = sensor noise / dark current (captured with lens cap on)
///   white = 100% reflectance reference (captured from a white calibration panel)
///
/// This normalizes each pixel to the 0–1 range, removing sensor-specific intensity
/// variations so that values represent actual surface reflectance.
/// </summary>
public static class HsiCalibration
{
    /// <summary>
    /// Scans the scene's directory for "dark" and "white" reference .hdr files.
    /// If both are found, loads them and applies reflectance calibration.
    /// Returns null if either reference is missing (calibration is skipped).
    /// </summary>
    public static async Task<HsiCube?> TryCalibrateAsync(string sceneHdrPath, HsiCube scene, CancellationToken ct = default)
    {
        var dir = Path.GetDirectoryName(sceneHdrPath)!;
        string? darkPath = null;
        string? whitePath = null;

        foreach (var file in Directory.GetFiles(dir, "*.hdr"))
        {
            var name = Path.GetFileNameWithoutExtension(file)
                .ToLowerInvariant();

            if (name.Contains("dark"))
                darkPath = file;
            else if (name.Contains("white"))
                whitePath = file;
        }

        if (darkPath == null || whitePath == null)
            return null;

        // Load both references in parallel
        var darkTask = HsiCubeLoader.LoadAsync(
            HsiHeaderParser.Parse(darkPath), ct: ct);
        var whiteTask = HsiCubeLoader.LoadAsync(
            HsiHeaderParser.Parse(whitePath), ct: ct);
        await Task.WhenAll(darkTask, whiteTask);
        var dark = darkTask.Result;
        var white = whiteTask.Result;

        return Calibrate(scene, dark, white);
    }

    /// <summary>
    /// Applies reflectance = (raw - dark) / (white - dark) to every pixel in every band.
    ///
    /// Each band is independent, so bands are calibrated in parallel across CPU cores.
    ///
    /// Two reference modes are supported:
    ///   - Full-frame: reference has the same dimensions as the scene (one dark/white value per pixel)
    ///   - Line reference: reference has a single line (one dark/white value per column,
    ///     shared across all rows). This is common when the reference is captured as a
    ///     single scan line from a push-broom sensor.
    /// </summary>
    private static HsiCube Calibrate(HsiCube raw, HsiCube dark, HsiCube white)
    {
        var header = raw.Header;
        var bands = header.Bands;
        var samples = header.Samples;
        var lines = header.Lines;
        var pixels = lines * samples;
        var lineRef = dark.Header.Lines != lines;

        var result = new float[bands * pixels];

        // Each band is calibrated independently — Parallel.For distributes bands across cores
        Parallel.For(0, bands, b =>
        {
            var rawBand = raw.GetBand(b);
            var darkBand = dark.GetBand(b);
            var whiteBand = white.GetBand(b);
            var offset = b * pixels;

            if (lineRef)
            {
                // Single-line reference: dark/white have one value per column (x).
                // Precompute 1/(white-dark) per column to avoid repeated division.
                Span<float> invDenom = stackalloc float[samples];
                Span<float> darkCol = stackalloc float[samples];
                for (var x = 0; x < samples; x++)
                {
                    var d = whiteBand[x] - darkBand[x];
                    invDenom[x] = d > 1f ? 1f / d : 0f;
                    darkCol[x] = darkBand[x];
                }

                // Apply: reflectance = (raw - dark) * (1 / (white - dark))
                for (var y = 0; y < lines; y++)
                {
                    var rowStart = y * samples;
                    for (var x = 0; x < samples; x++)
                        result[offset + rowStart + x] =
                            (rawBand[rowStart + x] - darkCol[x]) * invDenom[x];
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
                        ? (rawBand[i] - darkBand[i]) / d
                        : 0f;
                }
            }
        });

        return new HsiCube(header, result);
    }
}