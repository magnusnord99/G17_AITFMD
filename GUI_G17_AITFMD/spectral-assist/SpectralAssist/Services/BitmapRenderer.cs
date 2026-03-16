using System;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Converts HSI cube bands into displayable bitmaps.
/// Performs per-band min-max normalization to scale reflectance values to 0-255.
/// This scaling is for display only — the underlying cube data is not modified.
/// </summary>
public static class BitmapRenderer
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
                var value = Norm(band[i], min, range);
                ptr[i * 4 + 0] = value; // B
                ptr[i * 4 + 1] = value; // G
                ptr[i * 4 + 2] = value; // R
                ptr[i * 4 + 3] = 255;   // A (fully opaque)
            }
        }
        return bitmap;
    }

    /// <summary>
    /// Renders three bands as an RGB composite bitmap.
    /// Each band is independently normalized to preserve per-channel contrast.
    /// </summary>
    public static WriteableBitmap RgbToBitmap(HsiCube cube, int redBand, int greenBand, int blueBand)
    {
        var r = cube.GetBand(redBand);
        var g = cube.GetBand(greenBand);
        var b = cube.GetBand(blueBand);

        // Each channel is normalized independently so one bright band
        // doesn't wash out the others
        MinMax(r, out var rMin, out var rRange);
        MinMax(g, out var gMin, out var gRange);
        MinMax(b, out var bMin, out var bRange);

        var bitmap = CreateBitmap(cube);
        using var fb = bitmap.Lock();
        unsafe
        {
            // Pixel layout is BGRA (blue first) to match Bgra8888 format
            var ptr = (byte*)fb.Address;
            for (var i = 0; i < r.Length; i++)
            {
                ptr[i * 4 + 0] = Norm(b[i], bMin, bRange); // B
                ptr[i * 4 + 1] = Norm(g[i], gMin, gRange); // G
                ptr[i * 4 + 2] = Norm(r[i], rMin, rRange); // R
                ptr[i * 4 + 3] = 255;                       // A
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
    /// Scales a single reflectance value to a display byte (0-255).
    /// </summary>
    private static byte Norm(float value, float min, float range)
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
        if (range < 1e-6f) range = 1f; // Prevent division by zero
    }
}
