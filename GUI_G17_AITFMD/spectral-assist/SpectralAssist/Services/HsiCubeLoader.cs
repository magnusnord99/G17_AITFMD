using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Loads an ENVI binary data file and converts it to BSQ (Band Sequential) layout.
///
/// The pipeline is:
///   1. Read the raw binary file into memory
///   2. Cast the bytes to float[] based on the ENVI data type (uint16, float32, etc.)
///   3. Reorder the flat array from the file's interleave format into BSQ
///
/// BSQ means the data is stored as [band0_all_pixels, band1_all_pixels, ...],
/// which makes per-band operations (rendering, calibration) a simple contiguous slice.
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

    /// <summary>
    /// Converts raw bytes into a float[] in BSQ order.
    /// First casts bytes to floats, then reorders if the file isn't already BSQ.
    /// </summary>
    private static float[] ConvertToBsq(byte[] raw, HsiHeader header,
        IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var samples = header.Samples;
        var lines = header.Lines;
        var bands = header.Bands;
        var pixelsPerBand = samples * lines;
        var totalElements = samples * lines * bands;

        // Step 1: Reinterpret raw bytes as float[] (e.g. every 2 bytes -> one uint16 -> float)
        var source = CastSource(raw, header.HeaderOffset, totalElements, header.DataType);

        // Step 2: Reorder into BSQ if needed
        switch (header.Interleave.ToLowerInvariant())
        {
            case "bsq":
                // Already in BSQ layout — no reordering needed
                progress?.Report((1f, bands - 1));
                return source;

            case "bil":
                return ReorderBil(source, samples, lines, bands, pixelsPerBand, progress, ct);

            case "bip":
                return ReorderBip(source, samples, lines, bands, pixelsPerBand, progress, ct);

            default:
                throw new NotSupportedException($"Unknown interleave: {header.Interleave}");
        }
    }

    /// <summary>
    /// Reorders BIL (Band Interleaved by Line) to BSQ.
    ///
    /// BIL layout:  for each line, all bands are stored consecutively:
    ///   [line0_band0_samples, line0_band1_samples, ..., line1_band0_samples, ...]
    ///
    /// We copy one row of samples at a time into the correct band slice in the output.
    /// </summary>
    private static float[] ReorderBil(float[] source, int samples, int lines, int bands,
        int pixelsPerBand, IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var result = new float[source.Length];
        for (var b = 0; b < bands; b++)
        {
            ct.ThrowIfCancellationRequested();
            var dstBandOffset = b * pixelsPerBand;
            for (var l = 0; l < lines; l++)
            {
                var srcLineOffset = l * bands * samples + b * samples;
                var dstLineOffset = dstBandOffset + l * samples;
                Array.Copy(source, srcLineOffset, result, dstLineOffset, samples);
            }
            progress?.Report(((float)(b + 1) / bands, b));
        }
        return result;
    }

    /// <summary>
    /// Reorders BIP (Band Interleaved by Pixel) to BSQ.
    ///
    /// BIP layout: for each pixel, all bands are stored consecutively:
    ///   [pixel0_band0, pixel0_band1, ..., pixel1_band0, pixel1_band1, ...]
    ///
    /// This is the most scattered layout for per-band access, so we gather
    /// each band's values by striding through the source with step = bands.
    /// </summary>
    private static float[] ReorderBip(float[] source, int samples, int lines, int bands,
        int pixelsPerBand, IProgress<(float, int)>? progress, CancellationToken ct)
    {
        var result = new float[source.Length];
        for (var b = 0; b < bands; b++)
        {
            ct.ThrowIfCancellationRequested();
            var dstBandOffset = b * pixelsPerBand;
            for (var l = 0; l < lines; l++)
            {
                var srcLineOffset = l * samples * bands;
                var dstLineOffset = dstBandOffset + l * samples;
                for (var s = 0; s < samples; s++)
                    result[dstLineOffset + s] = source[srcLineOffset + s * bands + b];
            }
            progress?.Report(((float)(b + 1) / bands, b));
        }
        return result;
    }

    /// <summary>
    /// Reinterprets the raw byte array as float[] by casting based on the ENVI data type code.
    /// Uses MemoryMarshal.Cast for zero-copy reinterpretation of the byte buffer.
    /// </summary>
    private static float[] CastSource(byte[] raw, int offset, int count, int dataType)
    {
        var span = raw.AsSpan(offset);
        var result = new float[count];

        switch (dataType)
        {
            case 1: // byte
                for (var i = 0; i < count; i++)
                    result[i] = span[i];
                break;
            case 12: // uint16
                var u16 = MemoryMarshal.Cast<byte, ushort>(span);
                for (var i = 0; i < count; i++)
                    result[i] = u16[i];
                break;
            case 2: // int16
                var s16 = MemoryMarshal.Cast<byte, short>(span);
                for (var i = 0; i < count; i++)
                    result[i] = s16[i];
                break;
            case 4: // float32
                var f32 = MemoryMarshal.Cast<byte, float>(span);
                for (var i = 0; i < count; i++)
                    result[i] = f32[i];
                break;
            case 3: // int32
                var s32 = MemoryMarshal.Cast<byte, int>(span);
                for (var i = 0; i < count; i++)
                    result[i] = s32[i];
                break;
            case 5: // float64
                var f64 = MemoryMarshal.Cast<byte, double>(span);
                for (var i = 0; i < count; i++)
                    result[i] = (float)f64[i];
                break;
            default:
                throw new NotSupportedException($"ENVI data type not supported: {dataType}");
        }

        return result;
    }
}