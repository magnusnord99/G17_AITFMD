using System;
using System.Buffers.Binary;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

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
    
    // Cast the entire byte array to typed values ONCE
    var source = CastSource(raw, header.HeaderOffset, totalElements, header.DataType);
    var result = new float[totalElements];

    var interleave = header.Interleave.ToLower();

    for (var b = 0; b < bands; b++)
    {
        ct.ThrowIfCancellationRequested();

        for (var l = 0; l < lines; l++)
        for (var s = 0; s < samples; s++)
        {
            var srcIdx = interleave switch
            {
                "bsq" => b * pixelsPerBand + l * samples + s,
                "bil" => l * bands * samples + b * samples + s,
                "bip" => l * samples * bands + s * bands + b,
                _ => throw new NotSupportedException($"Unknown interleave: {header.Interleave}")
            };
            var dstIdx = b * pixelsPerBand + l * samples + s;
            result[dstIdx] = source[srcIdx];
        }

        progress?.Report(((float)(b + 1) / bands, b));
    }

    return result;
}

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

/*
 * 


    private static float[] ConvertToBsq_Old(byte[] rawData, HsiHeader header, IProgress<(float, int)>? progress = null,  CancellationToken ct = default)
    {
        var pixelsPerBand = header.Samples * header.Lines;
        var totalElements = header.Samples * header.Lines * header.Bands;
        var bytesPerElement = GetBytesPerElement(header.DataType);
        var swap = header.ByteOrder == 1 != !BitConverter.IsLittleEndian;
        
        var result = new float[totalElements];

        for (var b = 0; b < header.Bands; b++)
        {
            ct.ThrowIfCancellationRequested();
            for (var l = 0; l < header.Lines; l++)
            for (var s = 0; s < header.Samples; s++)
            {
                var sourceIndex = GetSourceIndex(header, b, l, s);
                var byteOffset = header.HeaderOffset + sourceIndex * bytesPerElement;
                var destinationIndex = b * pixelsPerBand + l * header.Samples + s;
                
                result[destinationIndex] = ReadValue(rawData, byteOffset, header.DataType, swap);
                progress?.Report(((float)(b + 1) / header.Bands, b));
            }
        }

        return result;
    }
    
    private static float ReadValue(byte[] raw, int offset, int dataType, bool swap)
    {
        // If no swap needed
        if (!swap) return ParseValue(raw.AsSpan(offset), dataType);
        
        // Perform swap based on dataType
        var size = GetBytesPerElement(dataType);
        Span<byte> tmp = stackalloc byte[8];
        raw.AsSpan(offset, size).CopyTo(tmp);
        tmp[..size].Reverse();
        return ParseValue(tmp, dataType);

    }

    private static float ParseValue(Span<byte> span, int dataType)
    {
        return dataType switch
        {
            1  => span[0],                                                  // byte
            2  => BinaryPrimitives.ReadInt16LittleEndian(span),             // int16
            3  => BinaryPrimitives.ReadInt32LittleEndian(span),             // int32
            4  => BinaryPrimitives.ReadSingleLittleEndian(span),            // float32
            5  => (float)BinaryPrimitives.ReadDoubleLittleEndian(span),     // float64
            12 => BinaryPrimitives.ReadUInt16LittleEndian(span),            // uint16
            13 => BinaryPrimitives.ReadUInt32LittleEndian(span),            // uint32
            14 => BinaryPrimitives.ReadInt64LittleEndian(span),             // int64
            15 => (float)BinaryPrimitives.ReadUInt64LittleEndian(span),     // uint64
            _ => throw new NotSupportedException($"ENVI data type not supported: {dataType}")
        };
    }
    
    private static int GetSourceIndex(HsiHeader header, int b, int l, int s)
    {
        return header.Interleave.ToLower() switch
        {
            "bsq" => b * header.Lines * header.Samples + l * header.Samples + s,
            "bil" => l * header.Bands * header.Samples + b * header.Samples + s,
            "bip" => l * header.Samples * header.Bands + s * header.Bands + b,
            _ => throw new NotSupportedException($"Unknown interleave: {header.Interleave}")
        };
    }
    
    private static int GetBytesPerElement(int dataType) => dataType switch
    {
        1  => 1,
        2  => 2,
        3  => 4,
        4  => 4,
        5  => 8,
        12 => 2,
        13 => 4,
        14 => 8,
        15 => 8,
        _ => throw new NotSupportedException($"ENVI data type not supported: {dataType}")
    };
     */
    
}