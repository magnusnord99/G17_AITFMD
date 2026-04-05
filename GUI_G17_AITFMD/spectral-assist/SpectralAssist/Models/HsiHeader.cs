using System;

namespace SpectralAssist.Models;

public class HsiHeader
{
    public string? DataFilePath { get; set; }
    public string Description { get; set; } = string.Empty;
    public int HeaderOffset { get; set; }

    public int Samples { get; set; }
    public int Lines { get; set; }
    public int Bands { get; set; }

    public string Interleave { get; set; } = string.Empty;
    public int DataType { get; set; }
    public int ByteOrder { get; set; }

    public int[] DefaultBands { get; set; } = [0, 0, 0];
    public string WavelengthUnit { get; set; } = string.Empty;
    public float[] WavelengthValues { get; set; } = [];

    /// <summary>
    /// Returns the band index whose wavelength is closest to the target (in nm).
    /// </summary>
    public int FindClosestBand(float targetNm)
    {
        var closestBand = 0;
        var closestDistance = float.MaxValue;

        for (var i = 0; i < WavelengthValues.Length; i++)
        {
            var distance = MathF.Abs(WavelengthValues[i] - targetNm);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestBand = i;
            }
        }

        return closestBand;
    }
}