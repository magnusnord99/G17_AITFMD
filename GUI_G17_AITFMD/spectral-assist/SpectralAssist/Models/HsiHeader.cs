
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
}