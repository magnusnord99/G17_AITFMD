namespace SpectralAssist.Tests.Utils;

internal static class GoldenFloatLoaderHWB
{
    public static FloatCubeHWB LoadCube(int lines, int samples, int bands, string fileName)
    {
        var baseDir = AppContext.BaseDirectory;
        var path = Path.Combine(baseDir, "Fixtures", "baseline_golden", fileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Golden fixture not found: {path}");

        var expected = lines * samples * bands * sizeof(float);
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expected)
            throw new InvalidDataException($"Expected {expected} bytes in {fileName}, got {bytes.Length}.");

        var floats = new float[lines * samples * bands];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return new FloatCubeHWB(lines, samples, bands, floats);
    }

    public static float MaxAbsDiff(FloatCubeHWB a, FloatCubeHWB b)
    {
        if (a.Lines != b.Lines || a.Samples != b.Samples || a.Bands != b.Bands)
            throw new ArgumentException("Shape mismatch.");
        var max = 0f;
        for (var i = 0; i < a.Data.Length; i++)
        {
            var d = Math.Abs(a.Data[i] - b.Data[i]);
            if (d > max) max = d;
        }

        return max;
    }
}