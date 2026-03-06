using System;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public static class BitmapRenderer
{
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
                var value = Norm(band[i], min, range);
                ptr[i * 4 + 0] = value;
                ptr[i * 4 + 1] = value;
                ptr[i * 4 + 2] = value;
                ptr[i * 4 + 3] = 255;
            }
        }
        return bitmap;
    }

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
            var ptr = (byte*)fb.Address;
            for (var i = 0; i < r.Length; i++)
            {
                ptr[i * 4 + 0] = Norm(r[i], bMin, bRange); // R
                ptr[i * 4 + 1] = Norm(g[i], gMin, gRange); // G
                ptr[i * 4 + 2] = Norm(b[i], rMin, rRange); // B
                ptr[i * 4 + 3] = 255;
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

    private static byte Norm(float value, float min, float range)
    {
        return (byte)(Math.Clamp((value - min) / range, 0f, 1f) * 255f);
    }

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
        if (range < 1e-6f) range = 1f;
    }
}
