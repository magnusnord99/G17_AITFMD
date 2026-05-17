using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
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
    public static Bitmap BandToBitmap(HsiCube cube, int bandIndex)
    {
        var band = cube.GetBand(bandIndex);
        MinMax(band, out var min, out var range);

        var stride = cube.Samples * 4;
        var pixels = new byte[cube.Lines * stride];
        for (var i = 0; i < band.Length; i++)
        {
            // Same value in R, G, B channels produces grayscale
            var value = NormalizeClamp(band[i], min, range);
            pixels[i * 4 + 0] = value; // B
            pixels[i * 4 + 1] = value; // G
            pixels[i * 4 + 2] = value; // R
            pixels[i * 4 + 3] = 255;   // A
        }

        return CreateOpaqueBitmap(pixels, cube.Samples, cube.Lines, stride);
    }
    
    /// <summary>
    /// Builds immutable <see cref="Bitmap"/> from a pre-filled BGRA byte buffer.
    /// </summary>
    private static Bitmap CreateOpaqueBitmap(byte[] pixels, int width, int height, int stride)
    {
        unsafe
        {
            fixed (byte* ptr = pixels)
            {
                return new Bitmap(
                    PixelFormat.Bgra8888,
                    AlphaFormat.Opaque,
                    (IntPtr)ptr,
                    new PixelSize(width, height),
                    new Vector(96, 96),
                    stride);
            }
        }
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
    public static Bitmap SyntheticRgbToBitmap(HsiCube cube, SyntheticRgbParameters parameters)
    {
        var wavelengths = cube.Header.WavelengthValues;
        var pixelCount = cube.PixelsPerBand;
        var nBands = cube.Bands;
        
        var weightR = new float[nBands];
        var weightG = new float[nBands];
        var weightB = new float[nBands];
        
        float sumR = 0, sumG = 0, sumB = 0;

        for (var b = 0; b < nBands; b++)
        {
            var wavelength = wavelengths[b];
            weightR[b] = Gaussian(wavelength, parameters.MuR, parameters.SigmaR);
            sumR += weightR[b];
            weightG[b] = Gaussian(wavelength, parameters.MuG, parameters.SigmaG);
            sumG += weightG[b];
            weightB[b] = Gaussian(wavelength, parameters.MuB, parameters.SigmaB);
            sumB += weightB[b];
        }
        
        Normalize(weightR, sumR);
        Normalize(weightG, sumG);
        Normalize(weightB, sumB);
        
        var rCh = ArrayPool<float>.Shared.Rent(pixelCount);
        var gCh = ArrayPool<float>.Shared.Rent(pixelCount);
        var bCh = ArrayPool<float>.Shared.Rent(pixelCount);

        try
        {
            Array.Clear(rCh, 0, pixelCount);
            Array.Clear(gCh, 0, pixelCount);
            Array.Clear(bCh, 0, pixelCount);

            // Skip bands where Gaussian weight is effectively zero in all three channels.
            var significantBands = new List<int>(nBands);
            for (var b = 0; b < nBands; b++)
                if (weightR[b] >= 1e-8f || weightG[b] >= 1e-8f || weightB[b] >= 1e-8f)
                    significantBands.Add(b);
            
            // Process the data in parallel by splitting the index range into chunks
            Parallel.ForEach(Partitioner.Create(0, pixelCount), range =>
            {
                for (var k = 0; k < significantBands.Count; k++)
                {
                    var bandIndex = significantBands[k];
                    var wRb = weightR[bandIndex];
                    var wGb = weightG[bandIndex];
                    var wBb = weightB[bandIndex];

                    var band = cube.GetBand(bandIndex);
                    for (var i = range.Item1; i < range.Item2; i++)
                    {
                        var v = band[i];
                        rCh[i] += v * wRb;
                        gCh[i] += v * wGb;
                        bCh[i] += v * wBb;
                    }
                }
            });

            // Normalize each channel independently for display
            MinMax(rCh.AsSpan(0, pixelCount), out var rMin, out var rRange);
            MinMax(gCh.AsSpan(0, pixelCount), out var gMin, out var gRange);
            MinMax(bCh.AsSpan(0, pixelCount), out var bMin, out var bRange);
            
            var stride = cube.Samples * 4;
            var pixels = new byte[cube.Lines * stride];
            Parallel.ForEach(Partitioner.Create(0, pixelCount), range =>
            {
                for (var i = range.Item1; i < range.Item2; i++)
                {
                    var off = i * 4;
                    pixels[off + 0] = NormalizeClamp(bCh[i], bMin, bRange); // B
                    pixels[off + 1] = NormalizeClamp(gCh[i], gMin, gRange); // G
                    pixels[off + 2] = NormalizeClamp(rCh[i], rMin, rRange); // R
                    pixels[off + 3] = 255;                                  // A
                }
            });

            return CreateOpaqueBitmap(pixels, cube.Samples, cube.Lines, stride);
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
        if (sum < 1e-12f) 
            return;
        
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
    /// Uses broad Gaussian bands to produce stain-consistent visualization.
    /// </summary>
    public static SyntheticRgbParameters HistologyBalanced => new(
        MuR: 610f, SigmaR: 25f,
        MuG: 560f, SigmaG: 25f,
        MuB: 460f, SigmaB: 25f
    );
    
    /// <summary>
    /// Synthetic RGB parameters from the HistologyHSI-GB dataset paper
    /// (Ortega et al., Scientific Data 11:681, 2024, doi:10.1038/s41597-024-03510-x).
    /// The paper uses σ values 0.08, 0.06, 0.04 applied to the spectral range of the data (400–1000 = 600nm)
    /// This produces visual output consistent with the paper's Fig. 5c.
    /// </summary>
    public static SyntheticRgbParameters HistologyPaperExact => new(
        MuR: 590f, SigmaR: 48f,    // 0.08 x 600
        MuG: 560f, SigmaG: 36f,    // 0.06 x 600
        MuB: 470f, SigmaB: 24f     // 0.04 x 600
    );
    
    /// <summary>
    /// High-contrast RGB composite for surgical HSI datasets.
    /// Uses narrow Gaussian bands to produce more realistic RGB colors.
    /// </summary>
    public static SyntheticRgbParameters DiagnosticHighContrast => new(
        MuR: 630f, SigmaR: 18f,
        MuG: 545f, SigmaG: 18f,
        MuB: 455f, SigmaB: 18f
    );
}