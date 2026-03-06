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
}
