using System;
using System.Collections.Generic;
using SpectralAssist.Models;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.Services;

/// <summary>
/// Manifest-driven preprocessing pipeline. Executes steps declared in
/// <see cref="PreprocessingConfig.Steps"/> (or default steps if none specified).
/// All operations work directly on BSQ <see cref="HsiCube"/>, zero format conversions.
/// </summary>
public static class PreprocessingService
{
    private static readonly List<string> DefaultSteps =
        ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"];

    /// <summary>
    /// Run the full pipeline from raw data (includes calibration step).
    /// </summary>
    public static PreprocessingResult Run(HsiCube raw, HsiCube dark, HsiCube white, PreprocessingConfig config)
    {
        var steps = config.Steps is { Count: > 0 } ? config.Steps : DefaultSteps;
        var current = raw;
        bool[]? mask = null;

        foreach (var step in steps)
        {
            switch (step)
            {
                case "calibrate":
                    current = Calibration.Apply(current, dark, white, config.CalibrationEpsilon);
                    break;
                case "clip":
                    ReflectanceClip.ApplyInPlace(current, config.ClipMin, config.ClipMax);
                    break;
                case "neighbor_average":
                    current = NeighborAverage.Apply(current, config.NeighborAverageWindow);
                    break;
                case "tissue_mask":
                    mask = TissueMask.BuildMask(current, config.ToTissueOptions());
                    break;
                case "band_average":
                    current = BandAverageReducer.Apply(
                        current, config.BandReduceOutBands, config.BandReduceStrategy);
                    break;
                default:
                    throw new NotSupportedException($"Unknown preprocessing step: '{step}'");
            }
        }

        return new PreprocessingResult(current, mask);
    }

    /// <summary>
    /// Run from an already-calibrated cube. The "calibrate" step is skipped even if
    /// present in the step list (calibration already happened at load time).
    /// </summary>
    public static PreprocessingResult RunFromCalibrated(HsiCube calibrated, PreprocessingConfig config) 
    {
        var steps = config.Steps is { Count: > 0 } ? config.Steps : DefaultSteps;

        // Clone so in-place clip doesn't mutate the cached calibrated cube
        var current = calibrated.Clone();
        bool[]? mask = null;

        foreach (var step in steps)
        {
            switch (step)
            {
                case "calibrate":
                    // Already calibrated — skip
                    break;
                case "clip":
                    ReflectanceClip.ApplyInPlace(current, config.ClipMin, config.ClipMax);
                    break;
                case "neighbor_average":
                    current = NeighborAverage.Apply(current, config.NeighborAverageWindow);
                    break;
                case "tissue_mask":
                    mask = TissueMask.BuildMask(current, config.ToTissueOptions());
                    break;
                case "band_average":
                    current = BandAverageReducer.Apply(
                        current, config.BandReduceOutBands, config.BandReduceStrategy);
                    break;
                default:
                    throw new NotSupportedException($"Unknown preprocessing step: '{step}'");
            }
        }
        
        return new PreprocessingResult(current, mask);
    }
}

/// <summary>Output from <see cref="PreprocessingService"/>.</summary>
/// <param name="cube">Preprocessed BSQ cube ready for ONNX inference.</param>
/// <param name="tissueMask">Bool list that tells tissue from background per pixel, ordered row-major, or null if no mask step was run.</param>
public readonly struct PreprocessingResult(HsiCube cube, bool[]? tissueMask)
{
    public HsiCube Cube { get; } = cube;
    public bool[]? TissueMask { get; } = tissueMask;
}