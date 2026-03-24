namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Chains calibration → clip → neighbor average (avg3) → optional band average to N bands.
/// Tissue masking is not included here — add when C# matches <c>tissue_mask.build_tissue_mask</c>.
/// </summary>
public static class BaselineSpectralPipeline
{
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
