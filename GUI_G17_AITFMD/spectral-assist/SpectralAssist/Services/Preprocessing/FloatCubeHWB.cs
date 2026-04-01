using System;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Float32 cube stored in HWB layout matching NumPy (H, W, B) with C-order ravel:
/// index = (line * Samples + sample) * Bands + band.
/// This matches <c>np.ndarray.reshape(-1)</c> on shape (H, W, B) in C order.
/// </summary>
public sealed class FloatCubeHWB
{
    public FloatCubeHWB(int lines, int samples, int bands, float[] data)
    {
        if (lines < 1 || samples < 1 || bands < 1)
            throw new ArgumentOutOfRangeException(nameof(lines), "Lines, samples and bands must be positive.");
        var expected = lines * samples * bands;
        if (data.Length != expected)
            throw new ArgumentException($"Data length {data.Length} != {lines}*{samples}*{bands} = {expected}.");
        Lines = lines;
        Samples = samples;
        Bands = bands;
        Data = data;
    }

    public int Lines { get; }
    public int Samples { get; }
    public int Bands { get; }
    public float[] Data { get; }

    public static int FlatIndex(int line, int sample, int band, int samples, int bands) =>
        (line * samples + sample) * bands + band;

    public float Get(int line, int sample, int band) =>
        Data[FlatIndex(line, sample, band, Samples, Bands)];

    public void Set(int line, int sample, int band, float value) =>
        Data[FlatIndex(line, sample, band, Samples, Bands)] = value;

    public FloatCubeHWB CloneEmpty() => new(Lines, Samples, Bands, new float[Data.Length]);

    public static FloatCubeHWB FromFlat(int lines, int samples, int bands, ReadOnlySpan<float> data)
    {
        var expected = lines * samples * bands;
        if (data.Length != expected)
            throw new ArgumentException($"Expected {expected} floats, got {data.Length}.");
        var copy = new float[expected];
        data.CopyTo(copy);
        return new FloatCubeHWB(lines, samples, bands, copy);
    }
}
