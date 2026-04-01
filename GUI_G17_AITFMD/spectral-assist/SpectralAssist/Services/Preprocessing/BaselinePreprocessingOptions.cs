namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Options aligned with Python <c>calibrateClip.calibrate_cube</c>, <c>clip_cube</c>,
/// <c>spectral_transform.reduce_bands_neighbor_average</c>, and <c>band_reduce.reduce_bands_by_avg</c>.
/// </summary>
public readonly struct BaselinePreprocessingOptions
{
    public BaselinePreprocessingOptions(
        float calibrationEpsilon = 1e-8f,
        float clipMin = 0f,
        float clipMax = 1f,
        int neighborAverageWindow = 3,
        int bandReduceOutBands = 16,
        string bandReduceStrategy = "crop")
    {
        CalibrationEpsilon = calibrationEpsilon;
        ClipMin = clipMin;
        ClipMax = clipMax;
        NeighborAverageWindow = neighborAverageWindow;
        BandReduceOutBands = bandReduceOutBands;
        BandReduceStrategy = bandReduceStrategy;
    }

    /// <summary>Matches Python <c>eps</c> in denominator: (white - dark + eps).</summary>
    public float CalibrationEpsilon { get; }

    public float ClipMin { get; }
    public float ClipMax { get; }

    /// <summary>Avg3 uses window 3 in Python.</summary>
    public int NeighborAverageWindow { get; }

    public int BandReduceOutBands { get; }

    /// <summary>"crop" or "uneven" — see <c>band_reduce._compute_bin_sizes</c>.</summary>
    public string BandReduceStrategy { get; }
}
