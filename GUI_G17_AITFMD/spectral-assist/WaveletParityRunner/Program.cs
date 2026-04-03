using SpectralAssist.Services.Preprocessing;

const float Tolerance = 1e-4f;

var fixturesDir = ResolveFixturesDir(args);
var cube = LoadCube(Path.Combine(fixturesDir, "cube_hwb_f32.bin"), 4, 4, 275);
var expected = LoadCube(Path.Combine(fixturesDir, "wavelet16_expect.bin"), 4, 4, 16);

var actual = WaveletReducer.ApplyApproxPaddedDb2(
    cube,
    targetBands: 16,
    level: null,
    mode: "periodization",
    padMode: "edge");

var diff = MaxAbsDiff(actual, expected);

Console.WriteLine($"Fixtures: {fixturesDir}");
Console.WriteLine($"Max abs diff: {diff}");
Console.WriteLine($"Tolerance:    {Tolerance}");

if (diff >= Tolerance)
{
    Console.Error.WriteLine("Wavelet parity FAILED.");
    return 1;
}

Console.WriteLine("Wavelet parity OK.");
return 0;

static string ResolveFixturesDir(string[] args)
{
    if (args.Length > 0)
        return Path.GetFullPath(args[0]);

    return Path.GetFullPath(
        Path.Combine(
            AppContext.BaseDirectory,
            "..", "..", "..", "..",
            "SpectralAssist.Tests",
            "Fixtures",
            "wavelet_golden"));
}

static FloatCubeHWB LoadCube(string path, int lines, int samples, int bands)
{
    if (!File.Exists(path))
        throw new FileNotFoundException($"Fixture not found: {path}");

    var expectedBytes = lines * samples * bands * sizeof(float);
    var bytes = File.ReadAllBytes(path);
    if (bytes.Length != expectedBytes)
    {
        throw new InvalidDataException(
            $"Expected {expectedBytes} bytes in {path}, got {bytes.Length}.");
    }

    var floats = new float[lines * samples * bands];
    Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
    return new FloatCubeHWB(lines, samples, bands, floats);
}

static float MaxAbsDiff(FloatCubeHWB a, FloatCubeHWB b)
{
    if (a.Lines != b.Lines || a.Samples != b.Samples || a.Bands != b.Bands)
        throw new ArgumentException("Shape mismatch between actual and expected cubes.");

    var max = 0f;
    for (var i = 0; i < a.Data.Length; i++)
    {
        var diff = Math.Abs(a.Data[i] - b.Data[i]);
        if (diff > max)
            max = diff;
    }

    return max;
}
