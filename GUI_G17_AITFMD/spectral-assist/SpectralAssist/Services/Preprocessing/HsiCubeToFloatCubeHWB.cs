using SpectralAssist.Models;

namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Konverterer <see cref="HsiCube"/> (BSQ) til <see cref="FloatCubeHWB"/> (HWB) for baseline-pipeline.
/// </summary>
public static class HsiCubeToFloatCubeHWB
{
    public static FloatCubeHWB FromHsiCube(HsiCube cube)
    {
        var lines = cube.Lines;
        var samples = cube.Samples;
        var bands = cube.Bands;
        var data = new float[lines * samples * bands];
        for (var band = 0; band < bands; band++)
        {
            var src = cube.GetBand(band);
            for (var line = 0; line < lines; line++)
            {
                var row = line * samples;
                for (var s = 0; s < samples; s++)
                {
                    var idx = FloatCubeHWB.FlatIndex(line, s, band, samples, bands);
                    data[idx] = src[row + s];
                }
            }
        }

        return new FloatCubeHWB(lines, samples, bands, data);
    }
}
