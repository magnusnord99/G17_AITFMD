using System;
using System.Collections.Generic;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Parity with Python <c>tissue_mask.build_tissue_mask</c> when
/// <c>method == "mean_std_percentile"</c> (mean + spectral std + percentiles + morphology).
/// </summary>
public static class TissueMaskMeanStdPercentile
{
    /// <summary>Build boolean tissue mask; result length = Lines × Samples (row-major).</summary>
    public static bool[] BuildMask(FloatCubeHWB cube, in TissueMaskMeanStdOptions options)
    {
        if (options.QMean is <= 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "q_mean must be in (0, 1).");
        if (options.QStd is <= 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "q_std must be in (0, 1).");

        var mean = ComputeMeanMap(cube);
        var std = ComputeStdMap(cube);
        var tMean = NumpyCompatibleQuantile.LinearQuantile(mean, options.QMean);
        var tStd = NumpyCompatibleQuantile.LinearQuantile(std, options.QStd);

        var h = cube.Lines;
        var w = cube.Samples;
        var n = h * w;
        var raw = new bool[n];
        for (var i = 0; i < n; i++)
            raw[i] = std[i] > tStd || mean[i] < tMean;

        return CleanMask(raw, h, w, options.MinObjectSize, options.MinHoleSize);
    }

    /// <summary>Mean over bands per pixel (NumPy: mean over axis 2, float32).</summary>
    public static float[] ComputeMeanMap(FloatCubeHWB cube)
    {
        var h = cube.Lines;
        var w = cube.Samples;
        var b = cube.Bands;
        var n = h * w;
        var map = new float[n];
        for (var line = 0; line < h; line++)
        {
            for (var s = 0; s < w; s++)
            {
                float sum = 0f;
                for (var band = 0; band < b; band++)
                    sum += cube.Get(line, s, band);
                map[line * w + s] = sum / b;
            }
        }

        return map;
    }

    /// <summary>Population std over bands (NumPy: std over axis 2, float32).</summary>
    public static float[] ComputeStdMap(FloatCubeHWB cube)
    {
        var h = cube.Lines;
        var w = cube.Samples;
        var bands = cube.Bands;
        var n = h * w;
        var map = new float[n];
        for (var line = 0; line < h; line++)
        {
            for (var s = 0; s < w; s++)
            {
                float sum = 0f;
                for (var band = 0; band < bands; band++)
                    sum += cube.Get(line, s, band);
                var mu = sum / bands;
                float acc = 0f;
                for (var band = 0; band < bands; band++)
                {
                    var d = cube.Get(line, s, band) - mu;
                    acc += d * d;
                }

                var variance = acc / bands;
                map[line * w + s] = MathF.Sqrt(variance);
            }
        }

        return map;
    }

    /// <summary>Matches Python <c>tissue_mask._clean_mask</c> with skimage fallback API (<c>min_size</c> / <c>area_threshold</c>).</summary>
    public static bool[] CleanMask(bool[] rawMask, int height, int width, int minObjectSize, int minHoleSize)
    {
        if (rawMask.Length != height * width)
            throw new ArgumentException("Mask length does not match H×W.");

        var fg = RemoveSmallObjects(rawMask, height, width, minObjectSize);
        return RemoveSmallHoles(fg, height, width, minHoleSize);
    }

    /// <summary><c>skimage.morphology.remove_small_objects</c> with connectivity 1 (4-neighbors), <c>min_size</c> semantics.</summary>
    internal static bool[] RemoveSmallObjects(bool[] mask, int height, int width, int minSize)
    {
        if (minSize <= 0)
            return (bool[])mask.Clone();

        var n = height * width;
        var result = (bool[])mask.Clone();
        var visited = new bool[n];
        var stack = new Stack<int>();

        for (var idx = 0; idx < n; idx++)
        {
            if (!mask[idx] || visited[idx])
                continue;

            stack.Clear();
            stack.Push(idx);
            visited[idx] = true;
            var comp = new List<int>();
            while (stack.Count > 0)
            {
                var cur = stack.Pop();
                comp.Add(cur);
                var line = cur / width;
                var sample = cur % width;
                TryNeighbor(line - 1, sample, height, width, mask, visited, stack);
                TryNeighbor(line + 1, sample, height, width, mask, visited, stack);
                TryNeighbor(line, sample - 1, height, width, mask, visited, stack);
                TryNeighbor(line, sample + 1, height, width, mask, visited, stack);
            }

            if (comp.Count < minSize)
            {
                foreach (var i in comp)
                    result[i] = false;
            }
        }

        return result;
    }

    private static void TryNeighbor(
        int line,
        int sample,
        int height,
        int width,
        bool[] mask,
        bool[] visited,
        Stack<int> stack)
    {
        if ((uint)line >= (uint)height || (uint)sample >= (uint)width)
            return;
        var j = line * width + sample;
        if (!mask[j] || visited[j])
            return;
        visited[j] = true;
        stack.Push(j);
    }

    /// <summary><c>skimage.morphology.remove_small_holes</c> (invert → remove_small_objects → invert).</summary>
    internal static bool[] RemoveSmallHoles(bool[] mask, int height, int width, int areaThreshold)
    {
        var n = height * width;
        var inv = new bool[n];
        for (var i = 0; i < n; i++)
            inv[i] = !mask[i];
        inv = RemoveSmallObjects(inv, height, width, areaThreshold);
        for (var i = 0; i < n; i++)
            inv[i] = !inv[i];
        return inv;
    }
}

/// <summary>Options aligned with Python <c>build_tissue_mask(..., method="mean_std_percentile")</c>.</summary>
public readonly struct TissueMaskMeanStdOptions
{
    public TissueMaskMeanStdOptions(
        float qMean = 0.5f,
        float qStd = 0.4f,
        int minObjectSize = 1000,
        int minHoleSize = 1000)
    {
        QMean = qMean;
        QStd = qStd;
        MinObjectSize = minObjectSize;
        MinHoleSize = minHoleSize;
    }

    public float QMean { get; }
    public float QStd { get; }
    public int MinObjectSize { get; }
    public int MinHoleSize { get; }
}

/// <summary>Default <c>np.quantile(..., method="linear")</c> (NumPy 2.x).</summary>
internal static class NumpyCompatibleQuantile
{
    public static float LinearQuantile(float[] values, float q)
    {
        if (values.Length == 0)
            throw new ArgumentException("Empty array.");
        var sorted = (float[])values.Clone();
        Array.Sort(sorted);
        var n = sorted.Length;
        if (n == 1)
            return sorted[0];

        var vi = (n - 1) * q;
        var low = (int)MathF.Floor(vi);
        var high = (int)MathF.Ceiling(vi);
        if (low == high)
            return sorted[low];
        return sorted[low] + (vi - low) * (sorted[high] - sorted[low]);
    }
}
