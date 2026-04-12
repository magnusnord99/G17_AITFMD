using System;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Wavelet spectral reducer using PyWavelets-compatible db2 decomposition
/// with approx_padded + periodization + edge padding.
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
    /// <list>
    /// <item>1) pad the spectral dimension to targetBands * 2^L</item>
    /// <item>2) run repeated db2 approximation steps</item>
    /// <item>3) keep only cA_L with exact length <paramref name="targetBands"/></item>
    /// </list>
    /// </summary>
    public static HsiCube Apply(
        HsiCube cube,
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
            throw new ArgumentException("Only mode='periodization' is supported.", nameof(mode));
        if (!string.Equals(padMode, "edge", StringComparison.Ordinal))
            throw new ArgumentException("Only padMode='edge' is supported.", nameof(padMode));

        var nBands = cube.Bands;
        var samples = cube.Samples;
        var lines = cube.Lines;
        var pixels = cube.PixelsPerBand;

        var minLevel = MinLevelForTarget(nBands, targetBands);
        var effectiveLevel = level ?? Math.Max(1, minLevel);
        if (effectiveLevel < minLevel)
        {
            throw new ArgumentException(
                $"level={effectiveLevel} is too small for bands={nBands} and targetBands={targetBands}. Need level >= {minLevel}.",
                nameof(level));
        }

        var paddedBands = targetBands * (1 << effectiveLevel);
        if (paddedBands < nBands)
            throw new InvalidOperationException("Internal error: padded length is smaller than input bands.");

        var output = new float[targetBands * pixels];

        // Process rows in parallel: each thread gets its own spectrum and pad buffers
        Parallel.For(0, lines, () => new float[nBands], (y, _, spectrum) =>
        {
            var rowStart = y * samples;

            for (var x = 0; x < samples; x++)
            {
                var px = rowStart + x;
                for (var b = 0; b < nBands; b++)
                    spectrum[b] = cube.GetBand(b)[px];

                // Pad and apply wavelet decomposition
                var current = EdgePadSpectrum(spectrum, paddedBands);
                for (var i = 0; i < effectiveLevel; i++)
                    current = DwtApproxOnceDb2(current);

                // Scatter: write reduced spectrum to BSQ output planes
                for (var b = 0; b < targetBands; b++)
                    output[b * pixels + px] = current[b];
            }

            return spectrum;
        }, _ => { });

        var header = new HsiHeader
        {
            Lines = cube.Lines,
            Samples = cube.Samples,
            Bands = targetBands,
            Interleave = "bsq",
        };

        return new HsiCube(header, output);
    }

    private static int MinLevelForTarget(int nBands, int targetBands)
    {
        if (nBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(nBands), "nBands must be > 0.");
        if (targetBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetBands), "targetBands must be > 0.");

        var ratio = Math.Max(1.0, nBands / (double)targetBands);
        return (int)Math.Ceiling(Math.Log(ratio, 2.0));
    }

    private static float[] EdgePadSpectrum(float[] spectrum, int paddedLength)
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

    private static float[] DwtApproxOnceDb2(float[] signal)
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
            var sum = 0.0;

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
