using System;
using SpectralAssist.Models;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.Services;

/// <summary>
/// Manifest-driven preprocessing pipeline.
///
/// Executes the ordered steps declared in the model package's <see cref="PreprocessingInfo"/>
/// to transform a raw or calibrated HSI cube into the format the ONNX model expects.
///
/// Two entry points:
/// <list>
/// <item><see cref="Run"/> — from raw data, runs every step including calibration.</item>
/// <item><see cref="RunFromCalibrated"/> — from a cached calibrated cube, skips calibration.
/// Clones the input first because some steps (e.g. clip) modify the cube in-place,
/// and the caller's cached cube must not be mutated.</item>
/// </list>
/// </summary>
public static class PreprocessingService
{
    /// <summary>
    /// Run the full pipeline from raw capture data.
    /// The manifest's step list should start with "calibrate".
    /// </summary>
    public static PreprocessingResult Run(
        HsiCube raw, HsiCube dark, HsiCube white, PreprocessingInfo preprocessing)
    {
        var cube = raw;
        bool[]? mask = null;

        foreach (var step in preprocessing.Steps)
            (cube, mask) = ApplyStep(step, cube, dark, white, mask, preprocessing.Params);

        return new PreprocessingResult(cube, mask);
    }

    /// <summary>
    /// Run from an already-calibrated cube (e.g. cached by <see cref="ImageLoadingService"/>).
    /// The "calibrate" step is skipped even if present in the manifest.
    ///
    /// The input is cloned because in-place steps (clip) would otherwise
    /// mutate the caller's cached cube, forcing a costly reload next time.
    /// </summary>
    public static PreprocessingResult RunFromCalibrated(
        HsiCube calibrated, PreprocessingInfo preprocessing)
    {
        var cube = calibrated.Clone();
        bool[]? mask = null;

        foreach (var step in preprocessing.Steps)
        {
            if (step == "calibrate") continue; // already done at load time
            (cube, mask) = ApplyStep(step, cube, null, null, mask, preprocessing.Params);
        }

        return new PreprocessingResult(cube, mask);
    }

    /// <summary>
    /// Executes a single preprocessing step, returning the (possibly replaced) cube
    /// and the (possibly updated) tissue mask.
    /// </summary>
    private static (HsiCube Cube, bool[]? Mask) ApplyStep(
        string step, HsiCube cube, HsiCube? dark, HsiCube? white,
        bool[]? mask, PreprocessingConfig config)
    {
        switch (step)
        {
            case "calibrate":
                return (Calibration.Apply(cube, dark!, white!, config.CalibrationEpsilon), mask);

            case "clip":
                ReflectanceClip.ApplyInPlace(cube, config.ClipMin, config.ClipMax);
                return (cube, mask);

            case "neighbor_average":
                return (NeighborAverage.Apply(cube, config.NeighborAverageWindow), mask);

            case "tissue_mask":
                return (cube, TissueMask.BuildMask(cube, config.ToTissueOptions()));

            case "band_average":
                return (BandAverageReducer.Apply(
                    cube, config.BandReduceOutBands, config.BandReduceStrategy), mask);

            default:
                throw new NotSupportedException($"Unknown preprocessing step: '{step}'");
        }
    }
}

/// <summary>Output of <see cref="PreprocessingService"/>.</summary>
/// <param name="cube">Preprocessed BSQ cube ready for ONNX inference.</param>
/// <param name="tissueMask">Per-pixel tissue/background flag (row-major), or null if no mask step was run.</param>
public readonly struct PreprocessingResult(HsiCube cube, bool[]? tissueMask)
{
    public HsiCube Cube { get; } = cube;
    public bool[]? TissueMask { get; } = tissueMask;
}