using System;
using System.Buffers;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Rendering;

/// <summary>
/// Converts HSI cube bands into displayable bitmaps.
/// Performs per-band min-max normalization to scale reflectance values to 0-255.
/// This scaling is for display only, the underlying cube data is not modified.
/// </summary>
public static class CubeRenderer
{
    /// <summary>
    /// Renders a single band as a grayscale bitmap.
    /// </summary>
    public static WriteableBitmap BandToBitmap(HsiCube cube, int bandIndex)
    {
        var band = cube.GetBand(bandIndex);
        MinMax(band, out var min, out var range);

        var bitmap = CreateBitmap(cube);
        using var fb = bitmap.Lock();
        unsafe
        {
            var ptr = (byte*)fb.Address;
            for (var i = 0; i < band.Length; i++)
            {
                // Same value in R, G, B channels produces grayscale
                var value = NormalizeClamp(band[i], min, range);
                ptr[i * 4 + 0] = value; // B
                ptr[i * 4 + 1] = value; // G
                ptr[i * 4 + 2] = value; // R
                ptr[i * 4 + 3] = 255; // A 
            }
        }

        return bitmap;
    }

    /// <summary>
    /// Renders three explicit band indices as an RGB composite bitmap.
    /// Each band is independently normalized to preserve per-channel contrast.
    /// </summary>
    public static WriteableBitmap RgbToBitmap(HsiCube cube, int redBand, int greenBand, int blueBand)
    {
        var r = cube.GetBand(redBand);
        var g = cube.GetBand(greenBand);
        var b = cube.GetBand(blueBand);

        MinMax(r, out var rMin, out var rRange);
        MinMax(g, out var gMin, out var gRange);
        MinMax(b, out var bMin, out var bRange);

        var bitmap = CreateBitmap(cube);
        using var fb = bitmap.Lock();
        unsafe
        {
            // Pixel layout is BGRA to match Bgra8888 format
            var ptr = (byte*)fb.Address;
            for (var i = 0; i < r.Length; i++)
            {
                ptr[i * 4 + 0] = NormalizeClamp(b[i], bMin, bRange); // B
                ptr[i * 4 + 1] = NormalizeClamp(g[i], gMin, gRange); // G
                ptr[i * 4 + 2] = NormalizeClamp(r[i], rMin, rRange); // R
                ptr[i * 4 + 3] = 255; // A
            }
        }

        return bitmap;
    }

    private static WriteableBitmap CreateBitmap(HsiCube cube)
    {
        return new WriteableBitmap(
            new PixelSize(cube.Samples, cube.Lines),
            new Vector(96, 96),
            PixelFormat.Bgra8888,
            AlphaFormat.Opaque);
    }

    /// <summary>
    /// Scales a single reflectance (0-1000+) value to a display byte (0-255).
    /// </summary>
    private static byte NormalizeClamp(float value, float min, float range)
    {
        return (byte)(Math.Clamp((value - min) / range, 0f, 1f) * 255f);
    }

    /// <summary>
    /// Finds the min value and value range of a band for normalization.
    /// Guards against zero-range bands (e.g. a constant-value band) to avoid division by zero.
    /// </summary>
    private static void MinMax(Span<float> band, out float min, out float range)
    {
        float low = float.MaxValue, high = float.MinValue;
        for (var i = 0; i < band.Length; i++)
        {
            if (band[i] < low) low = band[i];
            if (band[i] > high) high = band[i];
        }

        min = low;
        range = high - low;

        if (range < 1e-6f)
            range = 1f;
    }
    
    /// <summary>
    /// Renders a synthetic RGB composite by simulating the human eye spectral response
    /// using Gaussian-weighted integration across all spectral bands.
    /// </summary>
    public static WriteableBitmap SyntheticRgbToBitmap(HsiCube cube, SyntheticRgbParameters parameters)
    {
        var wavelengths = cube.Header.WavelengthValues;
        var pixelCount = cube.PixelsPerBand;
        var nBands = cube.Bands;
        
        var wR = new float[nBands];
        var wG = new float[nBands];
        var wB = new float[nBands];
        
        float sumR = 0, sumG = 0, sumB = 0;

        for (var b = 0; b < nBands; b++)
        {
            var wl = wavelengths[b];
            wR[b] = Gaussian(wl, parameters.MuR, parameters.SigmaR);
            sumR += wR[b];
            wG[b] = Gaussian(wl, parameters.MuG, parameters.SigmaG);
            sumG += wG[b];
            wB[b] = Gaussian(wl, parameters.MuB, parameters.SigmaB);
            sumB += wB[b];
        }
        
        Normalize(wR, sumR);
        Normalize(wG, sumG);
        Normalize(wB, sumB);
        
        var rCh = ArrayPool<float>.Shared.Rent(pixelCount);
        var gCh = ArrayPool<float>.Shared.Rent(pixelCount);
        var bCh = ArrayPool<float>.Shared.Rent(pixelCount);

        try
        {
            Array.Clear(rCh, 0, pixelCount);
            Array.Clear(gCh, 0, pixelCount);
            Array.Clear(bCh, 0, pixelCount);

            for (var bandIndex = 0; bandIndex < nBands; bandIndex++)
            {
                // Skip bands with negligible contribution to all channels
                if (wR[bandIndex] < 1e-8f && wG[bandIndex] < 1e-8f && wB[bandIndex] < 1e-8f)
                    continue;

                var band = cube.GetBand(bandIndex);
                for (var i = 0; i < pixelCount; i++)
                {
                    rCh[i] += band[i] * wR[bandIndex];
                    gCh[i] += band[i] * wG[bandIndex];
                    bCh[i] += band[i] * wB[bandIndex];
                }
            }

            // Normalize each channel independently for display
            MinMax(rCh.AsSpan(0, pixelCount), out var rMin, out var rRange);
            MinMax(gCh.AsSpan(0, pixelCount), out var gMin, out var gRange);
            MinMax(bCh.AsSpan(0, pixelCount), out var bMin, out var bRange);

            var bitmap = CreateBitmap(cube);
            using var fb = bitmap.Lock();
            unsafe
            {
                var ptr = (byte*)fb.Address;
                for (var i = 0; i < pixelCount; i++)
                {
                    ptr[i * 4 + 0] = NormalizeClamp(bCh[i], bMin, bRange); // B
                    ptr[i * 4 + 1] = NormalizeClamp(gCh[i], gMin, gRange); // G
                    ptr[i * 4 + 2] = NormalizeClamp(rCh[i], rMin, rRange); // R
                    ptr[i * 4 + 3] = 255; // A
                }
            }

            return bitmap;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rCh);
            ArrayPool<float>.Shared.Return(gCh);
            ArrayPool<float>.Shared.Return(bCh);
        }
    }
    
    /// <summary>
    /// Computes an unnormalized Gaussian weight for a value relative to a center.
    /// </summary>
    /// <param name="x">Input value (e.g., wavelength in nm).</param>
    /// <param name="mu">Center of the Gaussian (peak location).</param>
    /// <param name="sigma">Standard deviation controlling spread.</param>
    private static float Gaussian(float x, float mu, float sigma)
    {
        var z = (x - mu) / sigma;
        return MathF.Exp(-0.5f * z * z);
    }

    /// <summary>
    /// Normalizes an array of weights so their sum becomes 1.
    /// </summary>
    /// <param name="w">Array of weights to normalize in-place.</param>
    /// <param name="sum">Precomputed sum of the weights.</param>
    private static void Normalize(float[] w, float sum)
    {
        if (sum < 1e-12f) return;
        
        for (var i = 0; i < w.Length; i++)
            w[i] /= sum;
    }
}

/// <summary>
/// <para> Defines parameters for generating synthetic RGB composites from hyperspectral
/// data using Gaussian spectral weighting. </para>
/// <para> μ (Mu): The center wavelength (nm) which determines which spectral region contributes most.</para>
/// <para> σ (Sigma): The bandwidth (nm) which controls how wide the spectral contribution is.</para>
/// </summary>
public readonly record struct SyntheticRgbParameters(
    float MuR, float SigmaR,
    float MuG, float SigmaG,
    float MuB, float SigmaB)
{
    /// <summary>
    /// Balanced RGB composite for histology HSI datasets.
    /// Uses broader Gaussian bands to reduce noise and produce
    /// smooth, stain-consistent visualization.
    /// </summary>
    public static SyntheticRgbParameters HistologyBalanced => new(
        MuR: 610f, SigmaR: 25f,
        MuG: 560f, SigmaG: 25f,
        MuB: 460f, SigmaB: 25f
    );
    
    /// <summary>
    /// High-contrast RGB composite for surgical HSI datasets.
    /// Uses narrower Gaussian bands and shifted wavelengths to enhance
    /// spectral separability and improve visibility of perfused tissue.
    /// </summary>
    public static SyntheticRgbParameters DiagnosticHighContrast => new(
        MuR: 630f, SigmaR: 18f,
        MuG: 545f, SigmaG: 18f,
        MuB: 455f, SigmaB: 18f
    );
}