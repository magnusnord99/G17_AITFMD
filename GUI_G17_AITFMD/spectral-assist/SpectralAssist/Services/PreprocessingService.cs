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
                var hwb = HsiCubeToFloatCubeHWB.FromHsiCube(sceneCube);
                var reduced = WaveletReducer.ApplyApproxPaddedDb2(hwb,
                    config.BandReduceOutBands ?? throw new InvalidOperationException(
                        "Step 'wavelet' requires band_reduce_out_bands in manifest"));
                return (FloatCubeToHsiCube.ToHsiCube(reduced), mask);


            /* ToDO: Convert from HWB to BSQ straight?
             case "wavelet":
             return (WaveletReducer.Apply(cube, config.BandReduceOutBands), mask);
             */

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