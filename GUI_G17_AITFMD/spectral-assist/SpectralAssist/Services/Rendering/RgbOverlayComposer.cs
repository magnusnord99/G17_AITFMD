using System;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace SpectralAssist.Services.Rendering;

/// <summary>
/// Composites a BGRA heatmap overlay onto an opaque RGB background using a global layer opacity,
/// matching stacked <see cref="Avalonia.Controls.Image"/> controls (base + overlay with Opacity).
/// </summary>
public static class RgbOverlayComposer
{
    /// <summary>
    /// Alpha-blends <paramref name="overlay"/> onto <paramref name="background"/> per pixel.
    /// Effective blend factor is <c>(overlay_alpha / 255) * layerOpacity</c> where overlay alpha is 0 for transparent pixels.
    /// Both bitmaps must be Bgra8888 and the same dimensions.
    /// </summary>
    public static WriteableBitmap Compose(
        Bitmap background,
        Bitmap overlay,
        float layerOpacity)
    {
        ArgumentNullException.ThrowIfNull(background);
        ArgumentNullException.ThrowIfNull(overlay);

        var size = background.PixelSize;
        if (overlay.PixelSize != size)
            throw new ArgumentException("Background and overlay must have the same pixel dimensions.", nameof(overlay));

        if (layerOpacity is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(layerOpacity));

        var stride = size.Width * 4;
        var bgBytes = new byte[size.Height * stride];
        var olBytes = new byte[size.Height * stride];

        unsafe
        {
            fixed (byte* bgPtr = bgBytes)
            fixed (byte* olPtr = olBytes)
            {
                background.CopyPixels(new PixelRect(size), (IntPtr)bgPtr, bgBytes.Length, stride);
                overlay.CopyPixels(new PixelRect(size), (IntPtr)olPtr, olBytes.Length, stride);
            }
        }

        var result = new WriteableBitmap(
            size,
            background.Dpi,
            PixelFormat.Bgra8888,
            AlphaFormat.Opaque);

        using var outLock = result.Lock();

        unsafe
        {
            fixed (byte* bgPtr = bgBytes)
            fixed (byte* olPtr = olBytes)
            {
                var outPtr = (byte*)outLock.Address;
                var outStride = outLock.RowBytes;
                var height = size.Height;
                var width = size.Width;

                for (var y = 0; y < height; y++)
                {
                    var bgRow = y * stride;
                    var olRow = y * stride;
                    var outRow = y * outStride;
                    for (var x = 0; x < width; x++)
                    {
                        var bgO = bgRow + x * 4;
                        var olO = olRow + x * 4;
                        var outO = outRow + x * 4;
                        var oa = olPtr[olO + 3] / 255f;
                        var a = oa * layerOpacity;

                        if (a < 1e-6f)
                        {
                            outPtr[outO + 0] = bgPtr[bgO + 0];
                            outPtr[outO + 1] = bgPtr[bgO + 1];
                            outPtr[outO + 2] = bgPtr[bgO + 2];
                            outPtr[outO + 3] = 255;
                        }
                        else
                        {
                            outPtr[outO + 0] = (byte)(bgPtr[bgO + 0] * (1f - a) + olPtr[olO + 0] * a);
                            outPtr[outO + 1] = (byte)(bgPtr[bgO + 1] * (1f - a) + olPtr[olO + 1] * a);
                            outPtr[outO + 2] = (byte)(bgPtr[bgO + 2] * (1f - a) + olPtr[olO + 2] * a);
                            outPtr[outO + 3] = 255;
                        }
                    }
                }
            }
        }

        return result;
    }
}
