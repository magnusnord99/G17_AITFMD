using System.Collections.Generic;

namespace SpectralAssist.Models;

public record StrideOption(string Name, int Divisor)
{
    public override string ToString() => Name;
    public static StrideOption Default => Presets[0];

    public static readonly IReadOnlyList<StrideOption> Presets =
    [
        new("Model default", 0),
        new("0% overlap (fast)", 1),
        new("50% overlap (standard)", 2),
        new("75% overlap (fine)", 4),
        new("87.5% overlap (ultra)", 8),
        new("Pixel-perfect (slow)", -1),
    ];
}