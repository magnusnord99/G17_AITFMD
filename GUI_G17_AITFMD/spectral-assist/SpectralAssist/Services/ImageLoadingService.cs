using System;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;
using SpectralAssist.Services.Hsi;

namespace SpectralAssist.Services;

/// <summary>
/// Result from <see cref="ImageLoadingService.LoadAsync"/>.
/// </summary>
public readonly struct ImageLoadResult
{
    /// <summary>The loaded cube (calibrated if references were found, raw otherwise).</summary>
    public HsiCube Cube { get; init; }

    /// <summary>The calibrated cube for inference, or raw scene if no calibration.</summary>
    public HsiCube CalibratedCube { get; init; }

    /// <summary>Whether dark/white calibration was applied.</summary>
    public bool HasCalibration { get; init; }
}

/// <summary>
/// Encapsulates the full load-parse-calibrate pipeline for hyperspectral images.
/// </summary>
public class ImageLoadingService
{
    /// <summary>
    /// Loads an ENVI .hdr file, converts to BSQ, and applies reflectance calibration
    /// if dark/white reference files are found in the same folder.
    /// </summary>
    /// <param name="hdrPath">Path to the .hdr header file.</param>
    /// <param name="progress">Reports status messages and progress (0–1).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The loaded (and optionally calibrated) cube.</returns>
    /// <exception cref="InvalidOperationException">Thrown on band mismatch between scene and references.</exception>
    public static async Task<ImageLoadResult> LoadAsync(
        string hdrPath,
        IProgress<(string Status, double Progress)>? progress = null,
        CancellationToken ct = default)
    {
        // Step 1: Parse header
        progress?.Report(("Reading header...", 0));
        var header = HsiHeaderParser.Parse(hdrPath);
        progress?.Report(($"{header.Samples}x{header.Lines}x{header.Bands} bands, {header.Interleave.ToUpper()}", 0));

        // Step 2: Load binary data
        var loadProgress = new Progress<(float percent, int band)>(p =>
            progress?.Report(($"Loading image data... {p.percent:P0}", p.percent)));

        var scene = await HsiCubeLoader.LoadAsync(header, loadProgress, ct);

        // Step 3: Calibrate if references exist
        if (HsiCalibration.TryFindReferenceHdrPaths(hdrPath, out var darkHdr, out var whiteHdr))
        {
            progress?.Report(("Loading dark/white references...", 1));
            var darkTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(darkHdr!), ct: ct);
            var whiteTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(whiteHdr!), ct: ct);
            await Task.WhenAll(darkTask, whiteTask);

            var dark = darkTask.Result;
            var white = whiteTask.Result;

            if (scene.Bands != dark.Bands || scene.Bands != white.Bands)
            {
                throw new InvalidOperationException(
                    $"Band mismatch: scene {scene.Bands}, dark {dark.Bands}, white {white.Bands}");
            }

            var calibrated = HsiCalibration.ApplyReflectance(scene, dark, white);
            progress?.Report(("Calibration complete...", 1));

            return new ImageLoadResult
            {
                Cube = calibrated,
                CalibratedCube = calibrated,
                HasCalibration = true,
            };
        }

        progress?.Report(("Calibration skipped (no dark/white in folder)...", 1));
        return new ImageLoadResult
        {
            Cube = scene,
            CalibratedCube = scene,
            HasCalibration = false,
        };
    }
}