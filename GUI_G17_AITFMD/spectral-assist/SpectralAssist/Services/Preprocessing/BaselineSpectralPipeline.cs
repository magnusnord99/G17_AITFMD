using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Chains calibration → clip → neighbor average (avg3) → optional band average to N bands.
/// Tissue masking: see <see cref="TissueMaskMeanStdPercentile"/> (mean_std_percentile).
/// </summary>
public static class BaselineSpectralPipeline
{
    /// <summary>
    /// Kalibrering → clipping → nabogjennomsnitt (f.eks. 825 → 275 med vindu 3) → vevsmaske på avg3-kube
    /// → spektral nedskalering til 16 bånd. Maske beregnes på avg3-kuben (samme som Python-pipeline).
    /// </summary>
    public static BaselinePipelineResult RunThroughAvg16WithTissueMask(
        FloatCubeHWB raw,
        FloatCubeHWB dark,
        FloatCubeHWB white,
        BaselinePreprocessingOptions spectralOptions,
        TissueMaskMeanStdOptions tissueOptions)
    {
        var avg3 = RunThroughAvg3(raw, dark, white, spectralOptions);
        var tissueMask = TissueMaskMeanStdPercentile.BuildMask(avg3, tissueOptions);
        var cube16 = BandAverageReducer.Apply(
            avg3,
            spectralOptions.BandReduceOutBands,
            spectralOptions.BandReduceStrategy);
        return new BaselinePipelineResult(cube16, tissueMask);
    }

    public static FloatCubeHWB RunThroughAvg3(
        FloatCubeHWB raw,
        FloatCubeHWB dark,
        FloatCubeHWB white,
        BaselinePreprocessingOptions options)
    {
        var cal = ReflectanceCalibration.Apply(raw, dark, white, options.CalibrationEpsilon);
        var clipped = ReflectanceClip.Apply(cal, options.ClipMin, options.ClipMax);
        return SpectralNeighborAverage.Apply(clipped, options.NeighborAverageWindow);
    }

    public static FloatCubeHWB RunThroughAvg16(
        FloatCubeHWB raw,
        FloatCubeHWB dark,
        FloatCubeHWB white,
        BaselinePreprocessingOptions options)
    {
        var avg3 = RunThroughAvg3(raw, dark, white, options);
        return BandAverageReducer.Apply(avg3, options.BandReduceOutBands, options.BandReduceStrategy);
    }
}

/// <summary>Output from <see cref="BaselineSpectralPipeline.RunThroughAvg16WithTissueMask"/>.</summary>
public readonly struct BaselinePipelineResult
{
    public BaselinePipelineResult(FloatCubeHWB cube16Bands, bool[] tissueMask)
    {
        if (tissueMask.Length != cube16Bands.Lines * cube16Bands.Samples)
            throw new ArgumentException("Tissue mask length must equal Lines × Samples.");
        Cube16Bands = cube16Bands;
        TissueMask = tissueMask;
    }

    /// <summary>HWB-kube med <see cref="FloatCubeHWB.Bands"/> typisk 16 (for ONNX).</summary>
    public FloatCubeHWB Cube16Bands { get; }

    /// <summary>Én bool per piksel (rad-major: line × samples + sample), samme geometri som kuben.</summary>
    public bool[] TissueMask { get; }
}
