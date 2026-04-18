using System;
using System.IO;
using Avalonia;
using Avalonia.Media.Imaging;

namespace SpectralAssist.Services.Export;

/// <summary>PNG encoding and optional downscaling for PDF embedding.</summary>
public static class BitmapExportHelper
{
    public const int DefaultMaxEdgePixels = 2048;

    /// <summary>Scales down if width or height exceeds <paramref name="maxEdgePixels"/>; otherwise returns the original bitmap.</summary>
    public static Bitmap MaybeDownscale(Bitmap source, int maxEdgePixels = DefaultMaxEdgePixels)
    {
        var w = source.PixelSize.Width;
        var h = source.PixelSize.Height;
        if (w <= maxEdgePixels && h <= maxEdgePixels)
            return source;

        var scale = Math.Min((double)maxEdgePixels / w, (double)maxEdgePixels / h);
        var nw = Math.Max(1, (int)Math.Round(w * scale));
        var nh = Math.Max(1, (int)Math.Round(h * scale));
        return source.CreateScaledBitmap(new PixelSize(nw, nh), BitmapInterpolationMode.HighQuality);
    }

    public static byte[] ToPngBytes(Bitmap bitmap)
    {
        using var ms = new MemoryStream();
        bitmap.Save(ms);
        return ms.ToArray();
    }
}
