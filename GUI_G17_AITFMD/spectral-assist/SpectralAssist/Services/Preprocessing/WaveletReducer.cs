using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Draft implementation of the Python wavelet reducer for the specific mode used in this project:
/// db2 + approx_padded + periodization + edge padding.
///
/// Kept isolated on purpose: no GUI wiring, no DI registration, no pipeline integration yet.
/// </summary>
public static class WaveletReducer
{
    // PyWavelets "db2" decomposition low-pass filter.
    private static readonly double[] Db2DecLo =
    [
        -0.12940952255126037,
         0.22414386804185735,
         0.8365163037378079,
         0.48296291314453416,
    ];

    /// <summary>
    /// Apply the same high-level strategy as Python <c>reduce_cube_wavelet_approx_padded</c>:
    /// 1) pad the spectral dimension to targetBands * 2^L
    /// 2) run repeated db2 approximation steps
    /// 3) keep only cA_L with exact length <paramref name="targetBands"/>
    /// </summary>
    public static FloatCubeHWB ApplyApproxPaddedDb2(
        FloatCubeHWB cube,
        int targetBands = 16,
        int? level = null,
        string mode = "periodization",
        string padMode = "edge")
    {
        if (cube is null)
            throw new ArgumentNullException(nameof(cube));
        if (targetBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetBands), "targetBands must be > 0.");
        if (!string.Equals(mode, "periodization", StringComparison.Ordinal))
            throw new ArgumentException("Draft reducer currently supports only mode='periodization'.", nameof(mode));
        if (!string.Equals(padMode, "edge", StringComparison.Ordinal))
            throw new ArgumentException("Draft reducer currently supports only padMode='edge'.", nameof(padMode));

        var minLevel = MinLevelForTarget(cube.Bands, targetBands);
        var effectiveLevel = level ?? Math.Max(1, minLevel);
        if (effectiveLevel < minLevel)
        {
            throw new ArgumentException(
                $"level={effectiveLevel} is too small for bands={cube.Bands} and targetBands={targetBands}. Need level >= {minLevel}.",
                nameof(level));
        }

        var paddedBands = targetBands * (1 << effectiveLevel);
        if (paddedBands < cube.Bands)
            throw new InvalidOperationException("Internal error: padded length is smaller than input bands.");

        var output = new float[cube.Lines * cube.Samples * targetBands];
        var spectrum = new float[cube.Bands];

        for (var y = 0; y < cube.Lines; y++)
        {
            for (var x = 0; x < cube.Samples; x++)
            {
                for (var b = 0; b < cube.Bands; b++)
                    spectrum[b] = cube.Get(y, x, b);

                var current = EdgePadSpectrum(spectrum, paddedBands);
                for (var i = 0; i < effectiveLevel; i++)
                    current = DwtApproxOnceDb2(current);

                if (current.Length != targetBands)
                {
                    throw new InvalidOperationException(
                        $"Unexpected cA length after padding. Expected {targetBands}, got {current.Length}.");
                }

                for (var outBand = 0; outBand < targetBands; outBand++)
                {
                    var outIdx = FloatCubeHWB.FlatIndex(y, x, outBand, cube.Samples, targetBands);
                    output[outIdx] = current[outBand];
                }
            }
        }

        return new FloatCubeHWB(cube.Lines, cube.Samples, targetBands, output);
    }

    internal static int MinLevelForTarget(int nBands, int targetBands)
    {
        if (nBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(nBands), "nBands must be > 0.");
        if (targetBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetBands), "targetBands must be > 0.");

        var ratio = Math.Max(1.0, nBands / (double)targetBands);
        return (int)Math.Ceiling(Math.Log(ratio, 2.0));
    }

    internal static float[] EdgePadSpectrum(float[] spectrum, int paddedLength)
    {
        if (spectrum.Length == 0)
            throw new ArgumentException("Spectrum must be non-empty.", nameof(spectrum));
        if (paddedLength < spectrum.Length)
            throw new ArgumentOutOfRangeException(nameof(paddedLength), "Padded length must be >= input length.");

        var padded = new float[paddedLength];
        Array.Copy(spectrum, padded, spectrum.Length);

        var edgeValue = spectrum[^1];
        for (var i = spectrum.Length; i < padded.Length; i++)
            padded[i] = edgeValue;

        return padded;
    }

    internal static float[] DwtApproxOnceDb2(float[] signal)
    {
        if (signal.Length < 2 || signal.Length % 2 != 0)
            throw new ArgumentException("Signal length must be an even integer >= 2.", nameof(signal));

        var output = new float[signal.Length / 2];
        var filterLength = Db2DecLo.Length;

        for (var outIndex = 0; outIndex < output.Length; outIndex++)
        {
            // PyWavelets periodization for db2 aligns the low-pass filter
            // as a reversed filter with a -1 phase shift relative to the naive formulation.
            var start = (2 * outIndex) - 1;
            double sum = 0.0;

            for (var k = 0; k < filterLength; k++)
            {
                var wrapped = (start + k) % signal.Length;
                if (wrapped < 0)
                    wrapped += signal.Length;
                sum += signal[wrapped] * Db2DecLo[filterLength - 1 - k];
            }

            output[outIndex] = (float)sum;
        }

        return output;
    }
}
