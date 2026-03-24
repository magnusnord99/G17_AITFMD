namespace SpectralAssist.Services;

public record struct Color(byte R, byte G, byte B);

public static class ColorMaps
{
    /// <summary>
    /// Green (low probability) and Red (high probability).
    /// Good for binary healthy/tumor display.
    /// </summary>
    public static Color GreenRed(float probability)
    {
        return new Color(
            R: (byte)(probability * 255),
            G: (byte)((1f - probability) * 255),
            B: 0
        );
    }
}