using SpectralAssist.Models;
using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

public class BaselineFullPipelineTests
{
    [Fact]
    public void Hwb_to_HsiCube_Bsq_matches_pixel_access()
    {
        var data = new float[2 * 3 * 4];
        var idx = 0;
        for (var line = 0; line < 2; line++)
        for (var s = 0; s < 3; s++)
        for (var b = 0; b < 4; b++)
            data[idx++] = line * 100 + s * 10 + b;

        var cube = new FloatCubeHWB(2, 3, 4, data);
        var hsi = FloatCubeToHsiCube.ToHsiCube(cube);

        Assert.Equal(2, hsi.Lines);
        Assert.Equal(3, hsi.Samples);
        Assert.Equal(4, hsi.Bands);

        for (var line = 0; line < 2; line++)
        for (var s = 0; s < 3; s++)
        for (var b = 0; b < 4; b++)
        {
            var p = line * hsi.Samples + s;
            Assert.Equal(cube.Get(line, s, b), hsi.GetBand(b)[p]);
        }
    }

    [Fact]
    public void RunThroughAvg16WithTissueMask_produces_16_bands_and_mask_size()
    {
        var opts = new BaselinePreprocessingOptions(
            calibrationEpsilon: 1e-8f,
            clipMin: 0f,
            clipMax: 1f,
            neighborAverageWindow: 3,
            bandReduceOutBands: 16,
            bandReduceStrategy: "crop");

        var tissue = new TissueMaskMeanStdOptions(
            qMean: 0.5f,
            qStd: 0.4f,
            minObjectSize: 2,
            minHoleSize: 2);

        var raw = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_raw.bin");
        var dark = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_dark.bin");
        var white = GoldenFloatLoader.LoadCube(4, 4, 275, "chain275_white.bin");

        var result = BaselineSpectralPipeline.RunThroughAvg16WithTissueMask(raw, dark, white, opts, tissue);

        Assert.Equal(16, result.Cube16Bands.Bands);
        Assert.Equal(4 * 4, result.TissueMask.Length);

        var hsi = FloatCubeToHsiCube.ToHsiCube(result.Cube16Bands);
        Assert.Equal(16, hsi.Bands);
    }
}
