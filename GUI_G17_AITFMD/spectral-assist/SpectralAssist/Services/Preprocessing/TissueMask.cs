using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Builds a tissue mask from an <see cref="HsiCube"/> using mean+std percentile thresholds.
/// Uses band-plane accumulator pattern for cache-friendly BSQ traversal.
/// Includes morphological cleanup matching Python <c>tissue_mask._clean_mask</c>.
/// </summary>
public static class TissueMask
{
    /// <summary>Build boolean tissue mask; result length = Lines × Samples (row-major).</summary>
    public static bool[] BuildMask(HsiCube cube, in TissueMaskOptions options)
    {
        if (options.QMean is <= 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "q_mean must be in (0, 1).");
        if (options.QStd is <= 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "q_std must be in (0, 1).");

        var n = cube.PixelsPerBand;
        var bands = cube.Bands;

        // Single-pass mean + std using sum-of-squares identity: std = sqrt(E[x²] - E[x]²)
        // Parallelized across bands — each thread accumulates into its own buffer, merged at the end.
        var sum = new float[n];
        var sumSq = new float[n];

        Parallel.For(0, bands,
            () => (localSum: new float[n], localSumSq: new float[n]),
            (b, _, locals) =>
            {
                var band = cube.GetBand(b);
                for (var i = 0; i < n; i++)
                {
                    var v = band[i];
                    locals.localSum[i] += v;
                    locals.localSumSq[i] += v * v;
                }
                return locals;
            },
            locals =>
            {
                lock (sum)
                {
                    for (var i = 0; i < n; i++)
                    {
                        sum[i] += locals.localSum[i];
                        sumSq[i] += locals.localSumSq[i];
                    }
                }
            });

        // Derive mean and std from accumulated sums
        var invB = 1f / bands;
        var mean = new float[n];
        var std = new float[n];
        for (var i = 0; i < n; i++)
        {
            mean[i] = sum[i] * invB;
            std[i] = MathF.Sqrt(MathF.Max(0f, sumSq[i] * invB - mean[i] * mean[i]));
        }

        // Quantile thresholds
        var tMean = LinearQuantile(mean, options.QMean);
        var tStd = LinearQuantile(std, options.QStd);

        var raw = new bool[n];
        for (var i = 0; i < n; i++)
            raw[i] = std[i] > tStd || mean[i] < tMean;

        // Morphological cleanup
        return CleanMask(raw, cube.Lines, cube.Samples, options.MinObjectSize, options.MinHoleSize);
    }
    

    /// <summary>Matches Python <c>tissue_mask._clean_mask</c> with skimage fallback API.</summary>
    private static bool[] CleanMask(bool[] rawMask, int height, int width, int minObjectSize, int minHoleSize)
    {
        if (rawMask.Length != height * width)
            throw new ArgumentException("Mask length does not match H×W.");

        var fg = RemoveSmallObjects(rawMask, height, width, minObjectSize);
        return RemoveSmallHoles(fg, height, width, minHoleSize);
    }

    /// <summary><c>skimage.morphology.remove_small_objects</c> with connectivity 1 (4-neighbors).</summary>
    private static bool[] RemoveSmallObjects(bool[] mask, int height, int width, int minSize)
    {
        if (minSize <= 0)
            return (bool[])mask.Clone();

        var n = height * width;
        var result = (bool[])mask.Clone();
        var visited = new bool[n];
        var stack = new Stack<int>();
        var comp = new List<int>();  // Reused across components to avoid repeated allocation

        for (var idx = 0; idx < n; idx++)
        {
            if (!mask[idx] || visited[idx])
                continue;

            comp.Clear();
            stack.Clear();
            stack.Push(idx);
            visited[idx] = true;
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
        int line, int sample, int height, int width,
        bool[] mask, bool[] visited, Stack<int> stack)
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
    private static bool[] RemoveSmallHoles(bool[] mask, int height, int width, int areaThreshold)
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
    
    
    /// <summary>Default <c>np.quantile(..., method="linear")</c> (NumPy 2.x).</summary>
    private static float LinearQuantile(float[] values, float q)
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

/// <summary>Options aligned with Python <c>build_tissue_mask(..., method="mean_std_percentile")</c>.</summary>
public readonly struct TissueMaskOptions(
    float qMean = 0.5f,
    float qStd = 0.4f,
    int minObjectSize = 1000,
    int minHoleSize = 1000)
{
    public float QMean { get; } = qMean;
    public float QStd { get; } = qStd;
    public int MinObjectSize { get; } = minObjectSize;
    public int MinHoleSize { get; } = minHoleSize;
}