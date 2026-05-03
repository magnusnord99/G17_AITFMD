using System.Collections.Generic;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.Models;

public record DisplayOption(string Name, 
    DisplayMode DisplayMode, 
    SyntheticRgbParameters RgbParameters = default)
{
    public override string ToString() => Name;
    public static DisplayOption Default => Presets[0];

    public static readonly IReadOnlyList<DisplayOption> Presets =
    [
        new("Synthetic RGB - Histology", DisplayMode.SyntheticRgb, SyntheticRgbParameters.HistologyPaperExact),
        new("Synthetic RGB - Surgical", DisplayMode.SyntheticRgb, SyntheticRgbParameters.DiagnosticHighContrast),
        new("Spectral (single band)", DisplayMode.SpectralBand),
    ];
}

public enum DisplayMode
{
    SyntheticRgb,
    SpectralBand,
}