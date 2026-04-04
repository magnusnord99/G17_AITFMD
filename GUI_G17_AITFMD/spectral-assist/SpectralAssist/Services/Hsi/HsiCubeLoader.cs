using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Hsi;

/// <summary>
/// Loads raw binary HSI data and converts to BSQ float[] layout.
/// Uses Parallel.For on the band loop for BIL/BIP reordering
/// and on the type-cast loop for large arrays.
/// </summary>
public static class HsiCubeLoader
{
    public static async Task<HsiCube> LoadAsync(HsiHeader header, IProgress<(float, int)>? progress = null, CancellationToken ct = default)
    {
        if (header.DataFilePath == null)
            throw new FileNotFoundException("The data file was not found.");

        var rawData = await File.ReadAllBytesAsync(header.DataFilePath, ct);
        var data = await Task.Run(() => ConvertToBsq(rawData, header, progress, ct), ct);

        return new HsiCube(header, data);
    }

    private static float[] ConvertToBsq(byte[] raw, HsiHeader header,
        IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var samples = header.Samples;
        var lines = header.Lines;
        var bands = header.Bands;
        var pixelsPerBand = samples * lines;
        var totalElements = samples * lines * bands;

        var source = CastSourceParallel(raw, header.HeaderOffset, totalElements, header.DataType);

        switch (header.Interleave.ToLowerInvariant())
        {
            case "bsq":
                progress?.Report((1f, bands - 1));
                return source;

            case "bil":
                return ReorderBilParallel(source, samples, lines, bands, pixelsPerBand, progress, ct);

            case "bip":
                return ReorderBipParallel(source, samples, lines, bands, pixelsPerBand, progress, ct);

            default:
                throw new NotSupportedException($"Unknown interleave: {header.Interleave}");
        }
    }

    private static float[] ReorderBilParallel(float[] source, int samples, int lines, int bands,
        int pixelsPerBand, IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var result = new float[source.Length];
        var completed = 0;

        Parallel.For(0, bands, new ParallelOptions { CancellationToken = ct }, b =>
        {
            var dstBandOffset = b * pixelsPerBand;
            for (var l = 0; l < lines; l++)
            {
                var srcLineOffset = l * bands * samples + b * samples;
                var dstLineOffset = dstBandOffset + l * samples;
                Array.Copy(source, srcLineOffset, result, dstLineOffset, samples);
            }

            var done = Interlocked.Increment(ref completed);
            progress?.Report(((float)done / bands, b));
        });

        return result;
    }

    private static float[] ReorderBipParallel(float[] source, int samples, int lines, int bands,
        int pixelsPerBand, IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var result = new float[source.Length];
        var completed = 0;

        Parallel.For(0, bands, new ParallelOptions { CancellationToken = ct }, b =>
        {
            var dstBandOffset = b * pixelsPerBand;
            for (var l = 0; l < lines; l++)
            {
                var srcLineOffset = l * samples * bands;
                var dstLineOffset = dstBandOffset + l * samples;
                for (var s = 0; s < samples; s++)
                    result[dstLineOffset + s] = source[srcLineOffset + s * bands + b];
            }

            var done = Interlocked.Increment(ref completed);
            progress?.Report(((float)done / bands, b));
        });

        return result;
    }

    private static float[] CastSourceParallel(byte[] raw, int offset, int count, int dataType)
    {
        var result = new float[count];

        const int parallelThreshold = 500_000;
        if (count < parallelThreshold)
        {
            CastSequential(raw, offset, result, count, dataType);
            return result;
        }

        var bytesPerElement = dataType switch
        {
            1 => 1, 12 => 2, 2 => 2, 4 => 4, 3 => 4, 5 => 8,
            _ => throw new NotSupportedException($"ENVI data type not supported: {dataType}")
        };

        var chunkSize = Math.Max(count / Environment.ProcessorCount, 1024);
        var chunks = (count + chunkSize - 1) / chunkSize;

        Parallel.For(0, chunks, chunk =>
        {
            var start = chunk * chunkSize;
            var end = Math.Min(start + chunkSize, count);
            var localOffset = offset + start * bytesPerElement;
            var localSpan = raw.AsSpan(localOffset);

            switch (dataType)
            {
                case 1:
                    for (var i = start; i < end; i++)
                        result[i] = localSpan[i - start];
                    break;
                case 12:
                    var u16 = MemoryMarshal.Cast<byte, ushort>(localSpan);
                    for (var i = start; i < end; i++)
                        result[i] = u16[i - start];
                    break;
                case 2:
                    var s16 = MemoryMarshal.Cast<byte, short>(localSpan);
                    for (var i = start; i < end; i++)
                        result[i] = s16[i - start];
                    break;
                case 4:
                    var f32 = MemoryMarshal.Cast<byte, float>(localSpan);
                    for (var i = start; i < end; i++)
                        result[i] = f32[i - start];
                    break;
                case 3:
                    var s32 = MemoryMarshal.Cast<byte, int>(localSpan);
                    for (var i = start; i < end; i++)
                        result[i] = s32[i - start];
                    break;
                case 5:
                    var f64 = MemoryMarshal.Cast<byte, double>(localSpan);
                    for (var i = start; i < end; i++)
                        result[i] = (float)f64[i - start];
                    break;
            }
        });

        return result;
    }

    private static void CastSequential(byte[] raw, int offset, float[] result, int count, int dataType)
    {
        var span = raw.AsSpan(offset);
        switch (dataType)
        {
            case 1:
                for (var i = 0; i < count; i++) result[i] = span[i];
                break;
            case 12:
                var u16 = MemoryMarshal.Cast<byte, ushort>(span);
                for (var i = 0; i < count; i++) result[i] = u16[i];
                break;
            case 2:
                var s16 = MemoryMarshal.Cast<byte, short>(span);
                for (var i = 0; i < count; i++) result[i] = s16[i];
                break;
            case 4:
                var f32 = MemoryMarshal.Cast<byte, float>(span);
                for (var i = 0; i < count; i++) result[i] = f32[i];
                break;
            case 3:
                var s32 = MemoryMarshal.Cast<byte, int>(span);
                for (var i = 0; i < count; i++) result[i] = s32[i];
                break;
            case 5:
                var f64 = MemoryMarshal.Cast<byte, double>(span);
                for (var i = 0; i < count; i++) result[i] = (float)f64[i];
                break;
            default:
                throw new NotSupportedException($"ENVI data type not supported: {dataType}");
        }
    }
}