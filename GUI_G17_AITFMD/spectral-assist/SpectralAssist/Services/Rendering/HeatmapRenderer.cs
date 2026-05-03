using System;
using System.Buffers;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Rendering;

/// <summary>
/// Builds and renders classification heatmap overlays from inference results.
/// Separated into two stages for performance:
/// <list>
/// <item><see cref="BuildHeatmap"/>: expensive Gaussian-weighted accumulation (once per inference)</item>
/// <item><see cref="RenderHeatmap"/>: cheap colormap and threshold application (on display changes)</item>
/// </list>
/// </summary>
public static class HeatmapRenderer
{
    /// <summary>
    /// Builds a per-pixel probability heatmap using Gaussian-weighted accumulation of
    /// overlapping patch predictions. This is the expensive step — call once per inference
    /// result, then use <see cref="RenderHeatmap"/> to cheaply re-render with different
    /// thresholds or colormaps.
    /// </summary>
    public static float[] BuildHeatmap(
        ClassificationReport report,
        int width,
        int height,
        int targetClassIndex = 1)
    {
        var patchH = report.PatchH;
        var patchW = report.PatchW;
        var pixelCount = width * height;

        var heatmap = new float[pixelCount];
        var kernel = BuildGaussianKernel(patchH, patchW);

        // Rent a temporary weight buffer (per pixel weight)
        var weightSum = ArrayPool<float>.Shared.Rent(pixelCount);
        try
        {
            Array.Clear(weightSum, 0, pixelCount);

            // Gaussian-weighted accumulation of patch scores
            foreach (var pred in report.Predictions)
            {
                var score = pred.Probabilities[targetClassIndex];

                for (var dy = 0; dy < patchH; dy++)
                {
                    var py = pred.Y + dy;
                    if (py >= height) break;
                    var rowOffset = py * width;
                    var kernelRowOffset = dy * patchW;

                    for (var dx = 0; dx < patchW; dx++)
                    {
                        var px = pred.X + dx;
                        if (px >= width) break;

                        var idx = rowOffset + px;
                        var w = kernel[kernelRowOffset + dx];
                        heatmap[idx] += score * w;
                        weightSum[idx] += w;
                    }
                }
            }

            // Normalize in-place: heatmap[i] = weighted average probability
            for (var i = 0; i < pixelCount; i++)
            {
                if (weightSum[i] > 1e-6f)
                    heatmap[i] /= weightSum[i];
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(weightSum);
        }

        return heatmap;
    }
    
    /// <summary>
    /// Renders a cached per-pixel heatmap to a bitmap using the given colormap and threshold.
    /// </summary>
    public static Bitmap RenderHeatmap(
        float[] heatmap,
        int width,
        int height,
        Func<float, Color> colourMap,
        float threshold = 0f)
    {
        var stride = width * 4;
        var pixels = new byte[height * stride];

        for (var py = 0; py < height; py++)
        {
            var rowOffset = py * width;
            var bitmapRowOffset = py * stride;

            for (var px = 0; px < width; px++)
            {
                var avgProb = heatmap[rowOffset + px];
                if (avgProb < threshold || avgProb < 1e-6f) continue;

                var colour = colourMap(avgProb);
                var offset = bitmapRowOffset + px * 4;

                pixels[offset + 0] = colour.B; // B
                pixels[offset + 1] = colour.G; // G
                pixels[offset + 2] = colour.R; // R
                pixels[offset + 3] = 255;      // A
            }
        }

        return CreateUnpremulBitmap(pixels, width, height, stride);
    }

    /// <summary>
    /// Renders a horizontal gradient bar showing the active colormap from threshold to 1.0.
    /// Used as a legend for the classification overlay.
    /// </summary>
    public static Bitmap ColorBarLegend(
        Func<float, Color> colourMap,
        int width = 256,
        int height = 20,
        float threshold = 0f)
    {
        var stride = width * 4;
        var pixels = new byte[height * stride];

        for (var x = 0; x < width; x++)
        {
            // Map pixel position to probability range [threshold, 1.0]
            var prob = threshold + (float)x / (width - 1) * (1f - threshold);
            var colour = colourMap(prob);

            for (var y = 0; y < height; y++)
            {
                var offset = y * stride + x * 4;
                pixels[offset + 0] = colour.B; // B
                pixels[offset + 1] = colour.G; // G
                pixels[offset + 2] = colour.R; // R
                pixels[offset + 3] = 255;      // A
            }
        }

        return CreateUnpremulBitmap(pixels, width, height, stride);
    }

    /// <summary>
    /// Builds a Skia-native immutable <see cref="Bitmap"/> with unpremultiplied alpha
    /// from a pre-filled BGRA byte buffer. The ctor copies pixel data into Skia on
    /// construction, so <paramref name="pixels"/> may be collected afterwards.
    /// </summary>
    private static Bitmap CreateUnpremulBitmap(byte[] pixels, int width, int height, int stride)
    {
        unsafe
        {
            fixed (byte* ptr = pixels)
            {
                return new Bitmap(
                    PixelFormat.Bgra8888,
                    AlphaFormat.Unpremul,
                    (IntPtr)ptr,
                    new PixelSize(width, height),
                    new Vector(96, 96),
                    stride);
            }
        }
    }
    
    /// <summary>
    /// Builds a flattened 2D Gaussian kernel of size (patchH x patchW).
    /// Sigma defaults to patchH / 4, producing a smooth bell curve where
    /// center pixels have weight 1.0 and edge pixels fall off towards 0.
    /// </summary>
    private static float[] BuildGaussianKernel(int patchH, int patchW)
    {
        var kernel = new float[patchH * patchW];
        var sigmaH = patchH / 4.0f;
        var sigmaW = patchW / 4.0f;
        var centerY = (patchH - 1) / 2.0f;
        var centerX = (patchW - 1) / 2.0f;

        for (var y = 0; y < patchH; y++)
        {
            var dy = (y - centerY) / sigmaH;
            for (var x = 0; x < patchW; x++)
            {
                var dx = (x - centerX) / sigmaW;
                kernel[y * patchW + x] = MathF.Exp(-0.5f * (dy * dy + dx * dx));
            }
        }

        return kernel;
    }
}