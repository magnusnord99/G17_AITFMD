using System.Collections.Generic;

namespace SpectralAssist.Models;

public record DisplayOption(string Name, DisplayMode DisplayMode)
{
    public override string ToString() => Name;
    public static DisplayOption Default => Presets[0];

    public static readonly IReadOnlyList<DisplayOption> Presets =
    [
        new("Synthetic RGB (Gaussian)", DisplayMode.SyntheticRgb),
        new("RGB (nearest bands)", DisplayMode.NearestBandRgb),
        new("Spectral (single band)", DisplayMode.SpectralBand),
    ];
}

public enum DisplayMode
{
    SyntheticRgb,
    NearestBandRgb,
    SpectralBand
}
