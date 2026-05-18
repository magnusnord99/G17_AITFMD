using System;
using System.Threading;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Manifest-driven preprocessing pipeline.
/// Executes the ordered steps declared in the model package's <see cref="PreprocessingInfo"/>
/// to transform a raw or calibrated HSI cube into the format the ONNX model expects.
///
/// Two entry points:
/// <list>
/// <item><see cref="Run"/> from raw data, runs every step including calibration.</item>
/// <item><see cref="RunFromCalibrated"/> from a cached calibrated cube, skips calibration.
/// Clones the input first because some steps (e.g. clip) modify the cube in-place,
/// and the caller's cached cube must not be mutated.</item>
/// </list>
/// </summary>
public static class PreprocessingPipeline
{
    /// <summary>
    /// Runs the full preprocessing pipeline on raw capture data.
    /// The manifest's step list is executed in order, starting with "calibrate".
    /// Cancellation is checked between steps.
    /// </summary>
    /// <param name="raw">Raw, uncalibrated HSI cube.</param>
    /// <param name="dark">Dark reference cube.</param>men f
    /// <param name="white">White reference cube.</param>
    /// <param name="preprocessing">Preprocessing steps and parameters.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The processed cube and optional tissue mask.</returns>
    public static PreprocessingResult Run(HsiCube raw, HsiCube dark, HsiCube white, PreprocessingInfo preprocessing, 
        CancellationToken ct = default)
    {
        var cube = raw;
        bool[]? mask = null;

        foreach (var step in preprocessing.Steps)
        {
            ct.ThrowIfCancellationRequested();
            (cube, mask) = ApplyStep(step, cube, dark, white, mask, preprocessing.Params);
        }

        return new PreprocessingResult(cube, mask);
    }

    /// <summary>
    /// Runs preprocessing on an already‑calibrated cube. The calibration step is skipped.
    /// The cube is cloned to avoid mutating the caller's cached data.
    /// </summary>
    /// <param name="calibrated">A pre‑calibrated HSI cube, typically loaded from cache.</param>
    /// <param name="preprocessing">Preprocessing steps and parameters.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The processed cube and optional tissue mask.</returns>
    public static PreprocessingResult RunFromCalibrated(HsiCube calibrated, PreprocessingInfo preprocessing, 
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        var cube = calibrated.Clone();
        bool[]? mask = null;

        foreach (var step in preprocessing.Steps)
        {
            if (step == "calibrate") continue; // already done at load time
            ct.ThrowIfCancellationRequested();
            (cube, mask) = ApplyStep(step, cube, null, null, mask, preprocessing.Params);
        }

        return new PreprocessingResult(cube, mask);
    }

    /// <summary>
    /// Executes a single preprocessing step, returning the (possibly replaced) cube
    /// and the (possibly updated) tissue mask.
    /// </summary>
    private static (HsiCube Cube, bool[]? Mask) ApplyStep(
        string step, HsiCube sceneCube, HsiCube? darkCube, HsiCube? whiteCube,
        bool[]? mask, PreprocessingConfig config)
    {
        switch (step)
        {
            case "calibrate":
                return (Calibration.Apply(sceneCube, darkCube!, whiteCube!,
                    config.CalibrationEpsilon ?? throw new InvalidOperationException(
                        "Step 'calibrate' requires calibration_epsilon in manifest")), mask);

            case "clip":
                ReflectanceClip.ApplyInPlace(sceneCube,
                    config.ClipMin ?? throw new InvalidOperationException(
                        "Step 'clip' requires clip_min in manifest"),
                    config.ClipMax ?? throw new InvalidOperationException(
                        "Step 'clip' requires clip_max in manifest"));
                return (sceneCube, mask);

            case "neighbor_average":
                return (NeighborAverage.Apply(sceneCube,
                    config.NeighborAverageWindow ?? throw new InvalidOperationException(
                        "Step 'neighbor_average' requires neighbor_average_window in manifest")), mask);

            case "tissue_mask":
                return (sceneCube, TissueMask.BuildMask(sceneCube,
                    config.TissueMaskQMean ?? throw new InvalidOperationException(
                        "Step 'tissue_mask' requires tissue_mask_q_mean in manifest"),
                    config.TissueMaskQStd ?? throw new InvalidOperationException(
                        "Step 'tissue_mask' requires tissue_mask_q_std in manifest"),
                    config.TissueMaskMinObjectSize ?? throw new InvalidOperationException(
                        "Step 'tissue_mask' requires tissue_mask_min_object_size in manifest"),
                    config.TissueMaskMinHoleSize ?? throw new InvalidOperationException(
                        "Step 'tissue_mask' requires tissue_mask_min_hole_size in manifest"),
                    config.TissueMaskMethod ?? throw new InvalidOperationException(
                        "Step 'tissue_mask' requires tissue_mask_method in manifest")));

            case "band_average":
                return (BandAverageReducer.Apply(sceneCube,
                        config.BandReduceOutBands ??
                        throw new InvalidOperationException(
                            "Step 'band_average' requires band_reduce_out_bands in manifest"),
                        config.BandReduceStrategy ??
                        throw new InvalidOperationException(
                            "Step 'band_average' requires band_reduce_strategy in manifest")),
                    mask);

            case "wavelet":
                return (WaveletReducer.Apply(sceneCube,
                    config.BandReduceOutBands ?? throw new InvalidOperationException(
                        "Step 'wavelet' requires band_reduce_out_bands in manifest")), mask);
            
            default:
                throw new NotSupportedException($"Unknown preprocessing step: '{step}'");
        }
    }
}

/// <summary>Output of <see cref="PreprocessingPipeline"/>.</summary>
/// <param name="cube">Preprocessed BSQ cube ready for ONNX inference.</param>
/// <param name="tissueMask">Per-pixel tissue/background flag (row-major), or null if no mask step was run.</param>
public readonly struct PreprocessingResult(HsiCube cube, bool[]? tissueMask)
{
    public HsiCube Cube { get; } = cube;
    public bool[]? TissueMask { get; } = tissueMask;
}