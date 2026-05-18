using System;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Hsi;

/// <summary>
/// Result from <see cref="ImageLoader.LoadAsync"/>.
/// </summary>
public readonly struct ImageLoadResult
{
    /// <summary>The loaded cube (calibrated if references were found, raw otherwise).</summary>
    public HsiCube Cube { get; init; }

    /// <summary>Whether dark/white calibration was applied.</summary>
    public bool HasCalibration { get; init; }
}

/// <summary>
/// Coordinates the full load-parse-calibrate pipeline for hyperspectral images.
/// All heavy CPU work is dispatched off the UI thread.
/// </summary>
public class ImageLoader
{
    private const double HeaderEnd = 0.05;
    private const double SceneEndNoCal = 1.00;
    private const double SceneEndWithCal = 0.50;
    private const double ReferencesEnd = 0.70;
    private const double CalibrationEnd = 1.00;

    public static async Task<ImageLoadResult> LoadAsync(
        string hdrPath,
        IProgress<(string Status, double Progress)>? progress = null,
        Action<HsiHeader>? onHeaderParsed = null,
        CancellationToken ct = default)
    {
        //__ Step 1: Read header (0 - 5%) _______________________________________________
        progress?.Report(("Reading header...", 0));
        var header = HsiHeaderParser.Parse(hdrPath);
        onHeaderParsed?.Invoke(header);
        progress?.Report(("Reading header...", HeaderEnd));

        var willCalibrate = HsiCalibration.HasReferenceFiles(hdrPath);
        var sceneEnd = willCalibrate ? SceneEndWithCal : SceneEndNoCal;

        //__ Step 2: Load scene binary data (5 – 50% or 5 – 100%) ________________________
        var sceneLoadProgress = new Progress<(float percent, int band)>(p =>
        {
            var pct = HeaderEnd + p.percent * (sceneEnd - HeaderEnd);
            progress?.Report(("Loading image data…", pct));
        });
        var scene = await HsiCubeLoader.LoadAsync(header, sceneLoadProgress, ct);

        if (!willCalibrate)
        {
            progress?.Report(("Done", 1.0));
            return new ImageLoadResult { Cube = scene, HasCalibration = false };
        }

        //__ Step 3: Load dark/white references (50 – 70%) ____________________________
        progress?.Report(("Loading calibration references…", sceneEnd));
        var refs = await HsiCalibration.LoadReferencesAsync(hdrPath, ct);
        if (refs is not { } pair)
        {
            progress?.Report(("Done", 1.0));
            return new ImageLoadResult { Cube = scene, HasCalibration = false };
        }
        progress?.Report(("Loading calibration references…", ReferencesEnd));

        //__ Step 4: Apply reflectance (70 – 100%) ___________________________________
        progress?.Report(("Calibrating reflectance…", ReferencesEnd));
        var bandProgress = new Progress<float>(p =>
        {
            var pct = ReferencesEnd + p * (CalibrationEnd - ReferencesEnd);
            progress?.Report(("Calibrating reflectance…", pct));
        });

        var calibrated = await Task.Run(
            () => HsiCalibration.ApplyReflectance(scene, pair.Dark, pair.White,
                                                  bandProgress: bandProgress, ct: ct),
            ct);

        progress?.Report(("Done", 1.0));
        return new ImageLoadResult { Cube = calibrated, HasCalibration = true };
    }
}
