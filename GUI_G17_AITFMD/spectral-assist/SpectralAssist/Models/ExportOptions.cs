namespace SpectralAssist.Models;

public sealed record ExportOptions(
    float Overlay1Threshold,
    float Overlay2Threshold,
    string ColorMapName,
    float Opacity)
{
    public static ExportOptions Default => new(0.5f, 0.8f, "Green-Red", 0.5f);
}
