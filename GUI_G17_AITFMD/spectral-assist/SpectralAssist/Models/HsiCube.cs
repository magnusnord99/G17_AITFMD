using System;

namespace SpectralAssist.Models;

public class HsiCube(HsiHeader header, float[] data)
{
    public HsiHeader Header { get; } = header;

    public int Samples => Header.Samples;
    public int Lines => Header.Lines;
    public int Bands => Header.Bands;
    public int PixelsPerBand => Samples * Lines;

    public Span<float> GetBand(int band)
    {
        return data.AsSpan(band * PixelsPerBand, PixelsPerBand);
    }

    /// <summary>
    /// Returns the full spectral vector for one spatial pixel.
    /// </summary>
    public float[] GetSpectrumAt(int x, int y)
    {
        if (x < 0 || y < 0 || x >= Samples || y >= Lines)
            throw new ArgumentOutOfRangeException(
                $"Pixel ({x},{y}) is outside image bounds ({Samples}×{Lines})");

        var spectrum = new float[Bands];
        var pixelIndex = y * Samples + x;

        for (var b = 0; b < Bands; b++)
            spectrum[b] = GetBand(b)[pixelIndex];

        return spectrum;
    }

    /// <summary>Returns a deep copy of this cube with its own data buffer.</summary>
    public HsiCube Clone()
    {
        return new HsiCube(Header, (float[])data.Clone());
    }
    
    
    /// <summary>
    /// Extracts a spatial patch from the cube into a pre-allocated buffer.
    /// The buffer must have length Bands * patchH * patchW, and the result
    /// is written in BSQ order (bands, height, width).
    /// This is the allocation-free variant intended for hot loops such as
    /// patch-wise inference.
    /// </summary>
    public void ExtractPatchInto(int startX, int startY, int patchW, int patchH, Span<float> destination)
    {
        if (startX < 0 || startY < 0 || startX + patchW > Samples || startY + patchH > Lines)
            throw new ArgumentOutOfRangeException(
                $"Patch ({startX},{startY}) size ({patchW}×{patchH}) " +
                $"exceeds image bounds ({Samples}×{Lines})");

        if (destination.Length < Bands * patchH * patchW)
            throw new ArgumentException("Destination buffer is too small.", nameof(destination));

        for (var b = 0; b < Bands; b++)
        {
            var band = GetBand(b);
            var destOffset = b * patchH * patchW;

            for (var row = 0; row < patchH; row++)
            {
                // Copy one row of the patch at a time
                band.Slice((startY + row) * Samples + startX, patchW)
                    .CopyTo(destination.Slice(destOffset + row * patchW, patchW));
            }
        }
    }
    
    /// <summary>
    /// Convenience overload that allocates and returns a new buffer.
    /// Prefer <see cref="ExtractPatchInto"/> when calling repeatedly.
    /// </summary>
    public float[] ExtractPatch(int startX, int startY, int patchW, int patchH)
    {
        var result = new float[Bands * patchH * patchW];
        ExtractPatchInto(startX, startY, patchW, patchH, result);
        return result;
    }
    
}