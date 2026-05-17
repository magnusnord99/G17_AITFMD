namespace SpectralAssist.Models;

public sealed record ExportOptions(
    string ColorMapName,
    float OverlayOpacity,
    float Overlay1Threshold,
    float Overlay2Threshold)
{
    public static ExportOptions Default =>
        new("Viridis", 0.5f, 0.5f, 0.8f);

    public static ExportOptions FromOverlay(string colorMap, float opacity, float threshold) =>
        Default with { ColorMapName = colorMap, OverlayOpacity = opacity, Overlay1Threshold = threshold};
}