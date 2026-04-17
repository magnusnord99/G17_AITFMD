using System;
using System.Collections.Generic;

namespace SpectralAssist.Services.Rendering;

public record struct Color(byte R, byte G, byte B);

/// <summary>A collection of unique colormaps to be used for the overlay.</summary>
public static class ColorMaps
{
    /// <summary>All available colormaps keyed by display name.</summary>
    public static readonly Dictionary<string, Func<float, Color>> All = new()
    {
        ["Green-Red"] = GreenRed,
        ["Magma"] = Magma,
        ["Viridis"] = Viridis,
        ["Inferno"] = Inferno,
    };

    /// <summary>
    /// Green (low probability) to Red (high probability).
    /// Standard for binary healthy/tumor display.
    /// </summary>
    public static Color GreenRed(float probability)
    {
        return new Color(
            R: (byte)(probability * 255),
            G: (byte)((1f - probability) * 255),
            B: 0
        );
    }

    /// <summary>
    /// Black to Red to Yellow to White.
    /// Widely used for intensity/probability heatmaps in CT/MRI.
    /// </summary>
    private static Color Magma(float probability)
    {
        byte r, g, b;
        if (probability < 0.33f)
        {
            r = (byte)(probability / 0.33f * 255);
            g = 0;
            b = 0;
        }
        else if (probability < 0.66f)
        {
            r = 255;
            g = (byte)((probability - 0.33f) / 0.33f * 255);
            b = 0;
        }
        else
        {
            r = 255;
            g = 255;
            b = (byte)((probability - 0.66f) / 0.34f * 255);
        }

        return new Color(r, g, b);
    }

    /// <summary>
    /// Purple to Teal to Yellow. Perceptually uniform, colorblind-friendly.
    /// Modern standard in medical imaging.
    /// </summary>
    private static Color Viridis(float probability)
    {
        float r, g, b;
        if (probability < 0.25f)
        {
            var t = probability / 0.25f;
            r = Lerp(0.267f, 0.282f, t);
            g = Lerp(0.004f, 0.141f, t);
            b = Lerp(0.329f, 0.458f, t);
        }
        else if (probability < 0.5f)
        {
            var t = (probability - 0.25f) / 0.25f;
            r = Lerp(0.282f, 0.127f, t);
            g = Lerp(0.141f, 0.567f, t);
            b = Lerp(0.458f, 0.551f, t);
        }
        else if (probability < 0.75f)
        {
            var t = (probability - 0.5f) / 0.25f;
            r = Lerp(0.127f, 0.544f, t);
            g = Lerp(0.567f, 0.773f, t);
            b = Lerp(0.551f, 0.340f, t);
        }
        else
        {
            var t = (probability - 0.75f) / 0.25f;
            r = Lerp(0.544f, 0.993f, t);
            g = Lerp(0.773f, 0.906f, t);
            b = Lerp(0.340f, 0.144f, t);
        }

        return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }

    /// <summary>
    /// Black to Purple to Yellow. High-contrast heatmap.
    /// Good for highlighting regions of interest against dark backgrounds.
    /// </summary>
    private static Color Inferno(float probability)
    {
        float r, g, b;
        if (probability < 0.25f)
        {
            var t = probability / 0.25f;
            r = Lerp(0.001f, 0.258f, t);
            g = Lerp(0.001f, 0.039f, t);
            b = Lerp(0.014f, 0.406f, t);
        }
        else if (probability < 0.5f)
        {
            var t = (probability - 0.25f) / 0.25f;
            r = Lerp(0.258f, 0.565f, t);
            g = Lerp(0.039f, 0.050f, t);
            b = Lerp(0.406f, 0.342f, t);
        }
        else if (probability < 0.75f)
        {
            var t = (probability - 0.5f) / 0.25f;
            r = Lerp(0.565f, 0.871f, t);
            g = Lerp(0.050f, 0.318f, t);
            b = Lerp(0.342f, 0.098f, t);
        }
        else
        {
            var t = (probability - 0.75f) / 0.25f;
            r = Lerp(0.871f, 0.988f, t);
            g = Lerp(0.318f, 0.998f, t);
            b = Lerp(0.098f, 0.645f, t);
        }

        return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}