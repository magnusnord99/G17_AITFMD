using System;
using System.Collections.Generic;

namespace SpectralAssist.Services.Rendering;

/// <summary>
/// RGB color value with 8-bit channels
/// </summary>
public record struct Color(byte R, byte G, byte B);

/// <summary>
/// Curated colormaps for classification probability overlay rendering.
/// </summary>
public static class ColorMaps
{
    /// <summary>All available colormaps keyed by display name.</summary>
    public static readonly Dictionary<string, Func<float, Color>> All = new()
    {
        ["Viridis (default)"] = Viridis,
        ["Inferno (high-contrast)"] = Inferno,
        ["Green-Red (binary)"] = GreenRed,
        ["Blue-Red (diverging)"] = BuRd,
    };

    /// <summary>
    /// Sequential colormap: green (low) to red (high) with no in between colors.
    /// Traditional clinical binary convention where red marks abnormal/positive regions. 
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
    /// Sequential colormap: dark purple → blue → teal → green → yellow.
    /// Perceptually uniform and colorblind-safe. The modern default in
    /// libraries like matplotlib, plotly, and many scientific Python tools.
    /// </summary>
    private static Color Viridis(float probability)
    {
        float r, g, b;
        if (probability < 0.25f)
        {
            var t = probability / 0.25f;
            r = LinearInterpolation(0.267f, 0.282f, t);
            g = LinearInterpolation(0.004f, 0.141f, t);
            b = LinearInterpolation(0.329f, 0.458f, t);
        }
        else if (probability < 0.5f)
        {
            var t = (probability - 0.25f) / 0.25f;
            r = LinearInterpolation(0.282f, 0.127f, t);
            g = LinearInterpolation(0.141f, 0.567f, t);
            b = LinearInterpolation(0.458f, 0.551f, t);
        }
        else if (probability < 0.75f)
        {
            var t = (probability - 0.5f) / 0.25f;
            r = LinearInterpolation(0.127f, 0.544f, t);
            g = LinearInterpolation(0.567f, 0.773f, t);
            b = LinearInterpolation(0.551f, 0.340f, t);
        }
        else
        {
            var t = (probability - 0.75f) / 0.25f;
            r = LinearInterpolation(0.544f, 0.993f, t);
            g = LinearInterpolation(0.773f, 0.906f, t);
            b = LinearInterpolation(0.340f, 0.144f, t);
        }

        return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }

    /// <summary>
    /// Sequential colormap: black → dark purple → red → orange → light yellow.
    /// Perceptually uniform like Viridis but with a warmer, higher-contrast palette.
    /// </summary>
    private static Color Inferno(float probability)
    {
        float r, g, b;
        if (probability < 0.25f)
        {
            var t = probability / 0.25f;
            r = LinearInterpolation(0.001f, 0.258f, t);
            g = LinearInterpolation(0.001f, 0.039f, t);
            b = LinearInterpolation(0.014f, 0.406f, t);
        }
        else if (probability < 0.5f)
        {
            var t = (probability - 0.25f) / 0.25f;
            r = LinearInterpolation(0.258f, 0.565f, t);
            g = LinearInterpolation(0.039f, 0.050f, t);
            b = LinearInterpolation(0.406f, 0.342f, t);
        }
        else if (probability < 0.75f)
        {
            var t = (probability - 0.5f) / 0.25f;
            r = LinearInterpolation(0.565f, 0.871f, t);
            g = LinearInterpolation(0.050f, 0.318f, t);
            b = LinearInterpolation(0.342f, 0.098f, t);
        }
        else
        {
            var t = (probability - 0.75f) / 0.25f;
            r = LinearInterpolation(0.871f, 0.988f, t);
            g = LinearInterpolation(0.318f, 0.998f, t);
            b = LinearInterpolation(0.098f, 0.645f, t);
        }

        return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }
    
    /// Diverging colormap: blue (low) → white (midpoint) → red (high).
    /// The midpoint at 0.5 visually marks the classification decision boundary,
    /// making it easy to read whether a region is above or below the threshold.
    private static Color BuRd(float probability)
    {
        if (probability < 0.5f)
        {
            var t = probability / 0.5f;
            var r = LinearInterpolation(0.129f, 1f, t);
            var g = LinearInterpolation(0.400f, 1f, t);
            var b = LinearInterpolation(0.674f, 1f, t);
            return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }
        else
        {
            var t = (probability - 0.5f) / 0.5f;
            var r = LinearInterpolation(1f, 0.404f, t);
            var g = LinearInterpolation(1f, 0.000f, t);
            var b = LinearInterpolation(1f, 0.121f, t);
            return new Color((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }
    }

    /// <summary>
    /// Performs linear interpolation between two values.
    /// Returns <paramref name="a"/> when <paramref name="t"/> is 0,
    /// <paramref name="b"/> when <paramref name="t"/> is 1, and
    /// proportional values in between.
    /// </summary>
    /// <param name="a">The start value (returned when t = 0).</param>
    /// <param name="b">The end value (returned when t = 1).</param>
    /// <param name="t">Interpolation factor, typically in [0, 1].</param>
    /// <returns>The interpolated value at position <paramref name="t"/>.</returns>
    private static float LinearInterpolation(float a, float b, float t) => a + (b - a) * t;
}