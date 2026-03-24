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
    /// Extracts a spatial patch from the cube at the given position.
    /// Returns a flat array in BSQ order (bands, height, width),
    /// which is the layout ONNX models expect.
    /// </summary>
    public float[] ExtractPatch(int startX, int startY, int patchW, int patchH)
    {
        if (startX < 0 || startY < 0 || startX + patchW > Samples || startY + patchH > Lines) 
            throw new ArgumentOutOfRangeException(
                $"Patch ({startX},{startY}) size ({patchW}×{patchH}) " + 
                $"exceeds image bounds ({Samples}×{Lines})");

        var result = new float[Bands * patchH * patchW];

        for (var b = 0; b < Bands; b++)
        {
            var band = GetBand(b);  // full image row data for this band
            var destOffset = b * patchH * patchW;

            for (var y = 0; y < patchH; y++)
            {
                // Copy one row of the patch at a time
                // Source: row (startY + y) in full image, starting at column startX
                // Destination: row y in the patch
                band.Slice((startY + y) * Samples + startX, patchW)
                    .CopyTo(result.AsSpan(destOffset + y * patchW, patchW));
            }
        }

        return result;
    }
    
    
}
