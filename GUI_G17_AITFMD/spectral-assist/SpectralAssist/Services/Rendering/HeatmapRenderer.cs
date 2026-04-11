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
        ClassificationResult result,
        int width,
        int height,
        int targetClassIndex = 1)
    {
        var patchH = result.PatchH;
        var patchW = result.PatchW;
        var pixelCount = width * height;

        var heatmap = new float[pixelCount];
        var kernel = BuildGaussianKernel(patchH, patchW);

        // Rent a temporary weight buffer (per pixel weight)
        var weightSum = ArrayPool<float>.Shared.Rent(pixelCount);
        try
        {
            Array.Clear(weightSum, 0, pixelCount);

            // Gaussian-weighted accumulation of patch scores
            foreach (var pred in result.Predictions)
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
    public static WriteableBitmap RenderHeatmap(
        float[] heatmap,
        int width,
        int height,
        Func<float, Color> colourMap,
        float threshold = 0f)
    {
        var bitmap = CreateOverlayBitmap(width, height);
        using var buffer = bitmap.Lock();
        unsafe
        {
            var ptr = (byte*)buffer.Address;
            var stride = buffer.RowBytes;

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

                    ptr[offset + 0] = colour.B; // B
                    ptr[offset + 1] = colour.G; // G
                    ptr[offset + 2] = colour.R; // R
                    ptr[offset + 3] = 255;       // A
                }
            }
        }

        return bitmap;
    }
    
    /// <summary>
    /// Renders a horizontal gradient bar showing the active colormap from threshold to 1.0.
    /// Used as a legend for the classification overlay.
    /// </summary>
    public static WriteableBitmap ColorBarLegend(
        Func<float, Color> colourMap,
        int width = 256,
        int height = 20,
        float threshold = 0f)
    {
        var bitmap = new WriteableBitmap(
            new PixelSize(width, height),
            new Vector(96, 96),
            PixelFormat.Bgra8888,
            AlphaFormat.Unpremul);

        using var buffer = bitmap.Lock();
        unsafe
        {
            var ptr = (byte*)buffer.Address;
            var stride = buffer.RowBytes;

            for (var x = 0; x < width; x++)
            {
                // Map pixel position to probability range [threshold, 1.0]
                var prob = threshold + (float)x / (width - 1) * (1f - threshold);
                var colour = colourMap(prob);

                for (var y = 0; y < height; y++)
                {
                    var offset = y * stride + x * 4;
                    ptr[offset + 0] = colour.B; // B
                    ptr[offset + 1] = colour.G; // G
                    ptr[offset + 2] = colour.R; // R
                    ptr[offset + 3] = 255; // A
                }
            }
        }

        return bitmap;
    }
    
    /// <summary>
    /// Creates a bitmap for overlay rendering with unpremultiplied alpha,
    /// so uncovered pixels remain fully transparent.
    /// </summary>
    private static WriteableBitmap CreateOverlayBitmap(int width, int height)
    {
        return new WriteableBitmap(
            new PixelSize(width, height),
            new Vector(96, 96),
            PixelFormat.Bgra8888,
            AlphaFormat.Unpremul);
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