using System.Text.Json;
using System.Text.Json.Serialization;

namespace SpectralAssist.Services.Library;

public static class LibraryJson
{
    public static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
    };
}