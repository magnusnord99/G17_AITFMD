using System;
using System.IO;
using SpectralAssist.Models;

namespace SpectralAssist.Tests;

/// <summary>
/// Loads golden-file fixtures (Python-exported float32 binaries) as BSQ <see cref="HsiCube"/>.
/// Golden files are stored in HWB C-order (NumPy row-major), so this loader converts to BSQ.
/// </summary>
internal static class GoldenFloatLoader
{
    /// <summary>
    /// Loads a golden .bin file (HWB C-order) and returns a BSQ <see cref="HsiCube"/>.
    /// </summary>
    public static HsiCube LoadCube(int lines, int samples, int bands, string fileName)
    {
        var baseDir = AppContext.BaseDirectory;
        var path = Path.Combine(baseDir, "Fixtures", "baseline_golden", fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Golden fixture not found: {path}");

        var expected = lines * samples * bands * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");

        // Load as HWB (Python C-order)
        var hwb = new float[lines * samples * bands];
        Buffer.BlockCopy(bytes, 0, hwb, 0, bytes.Length);

        // Convert HWB → BSQ
        var bsq = ConvertHwbToBsq(hwb, lines, samples, bands);

        var header = new HsiHeader
        {
            Lines = lines,
            Samples = samples,
            Bands = bands,
            Interleave = "bsq",
        };
        return new HsiCube(header, bsq);
    }

    /// <summary>
    /// Loads a golden .bin file from a specific subfolder as BSQ <see cref="HsiCube"/>.
    /// </summary>
    public static HsiCube LoadCubeFromSubDir(int lines, int samples, int bands, string subDir, string fileName)
    {
        var baseDir = AppContext.BaseDirectory;
        var path = Path.Combine(baseDir, "Fixtures", subDir, fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Golden fixture not found: {path}");

        var expected = lines * samples * bands * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");

        var hwb = new float[lines * samples * bands];
        Buffer.BlockCopy(bytes, 0, hwb, 0, bytes.Length);

        var bsq = ConvertHwbToBsq(hwb, lines, samples, bands);

        var header = new HsiHeader
        {
            Lines = lines,
            Samples = samples,
            Bands = bands,
            Interleave = "bsq",
        };
        return new HsiCube(header, bsq);
    }

    /// <summary>Computes max absolute difference between two BSQ cubes.</summary>
    public static float MaxAbsDiff(HsiCube a, HsiCube b)
    {
        if (a.Lines != b.Lines || a.Samples != b.Samples || a.Bands != b.Bands)
            throw new ArgumentException("Shape mismatch.");
        var max = 0f;
        for (var band = 0; band < a.Bands; band++)
        {
            var aBand = a.GetBand(band);
            var bBand = b.GetBand(band);
            for (var i = 0; i < aBand.Length; i++)
            {
                var d = Math.Abs(aBand[i] - bBand[i]);
                if (d > max) max = d;
            }
        }
        return max;
    }

    /// <summary>
    /// HWB C-order (line, sample, band) → BSQ (band-major, each band is row-major H×W).
    /// </summary>
    private static float[] ConvertHwbToBsq(float[] hwb, int lines, int samples, int bands)
    {
        var plane = lines * samples;
        var bsq = new float[bands * plane];
        for (var line = 0; line < lines; line++)
        {
            for (var s = 0; s < samples; s++)
            {
                var hwbBase = (line * samples + s) * bands;
                var pixelIdx = line * samples + s;
                for (var b = 0; b < bands; b++)
                    bsq[b * plane + pixelIdx] = hwb[hwbBase + b];
            }
        }
        return bsq;
    }
}