using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

public class ReflectanceCalibrationTests
{
    [Fact]
    public void Line_reference_broadcasts_row0_to_match_full_frame_when_refs_are_one_line()
    {
        const float eps = 1e-8f;
        // 2×2 scene, 1 band
        var raw = new FloatCubeHWB(2, 2, 1, [10f, 20f, 30f, 40f]);
        var dark1 = new FloatCubeHWB(1, 2, 1, [1f, 2f]);
        var white1 = new FloatCubeHWB(1, 2, 1, [5f, 6f]);

        var darkFull = new FloatCubeHWB(2, 2, 1, [1f, 2f, 1f, 2f]);
        var whiteFull = new FloatCubeHWB(2, 2, 1, [5f, 6f, 5f, 6f]);

        var expected = ReflectanceCalibration.Apply(raw, darkFull, whiteFull, eps);
        var actual = ReflectanceCalibration.Apply(raw, dark1, white1, eps);

        Assert.Equal(expected.Data, actual.Data);
    }
}
