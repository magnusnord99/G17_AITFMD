using System;
using System.Collections.Generic;
using System.IO;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Inference;
using SpectralAssist.Services.Preprocessing;
using Xunit;

namespace SpectralAssist.Tests;

/*

/// <summary>
/// Tests for <see cref="ModelPackageValidator"/>: verifies that the smoke-test
/// correctly compares C# preprocessing output against Python golden data.
/// Golden data is now generated using the BSQ pipeline (same path the validator uses).
/// </summary>
public class ModelPackageValidationTests : IDisposable
{
    private readonly string _tempDir;

    public ModelPackageValidationTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"spectral_val_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    [Fact]
    public void Skipped_when_no_validation_folder()
    {
        var manifest = MakeManifest(includePrep: false, includeValidation: false);
        var result = ModelPackageValidator.Validate(_tempDir, manifest, session: null!);
        Assert.True(result.Passed);
        Assert.Contains("skipped", result.Summary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Skipped_when_manifest_has_no_validation_section()
    {
        var valDir = Path.Combine(_tempDir, "validation");
        Directory.CreateDirectory(valDir);

        var manifest = MakeManifest(includePrep: false, includeValidation: false);
        var result = ModelPackageValidator.Validate(_tempDir, manifest, session: null!);
        Assert.True(result.Passed);
        Assert.Contains("skipped", result.Summary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Preprocessing_passes_with_matching_golden_data()
    {
        // Create a small deterministic cube and run through BSQ pipeline,
        // then save the output as the "expected" golden data.
        const int h = 4, w = 4, bandsIn = 9;
        var prep = DefaultPrep();

        var raw = MakeBsqCube(h, w, bandsIn, seed: 42);
        var dark = MakeZeroBsqCube(h, w, bandsIn);
        var white = MakeOnesBsqCube(h, w, bandsIn);

        // Run through the BSQ pipeline (same path the validator uses)
        var pipeResult = PreprocessingService.Run(raw, dark, white, prep);
        var outBands = pipeResult.Cube.Bands;

        // Extract BSQ data for writing
        var cubeData = pipeResult.Cube.ExtractPatch(0, 0, w, h);

        // Save as golden files (simulating what Python export would produce)
        var valDir = Path.Combine(_tempDir, "validation");
        Directory.CreateDirectory(valDir);
        WriteF32Bin(Path.Combine(valDir, "ref_raw.bin"), ExtractAllData(raw));
        WriteF32Bin(Path.Combine(valDir, "ref_dark.bin"), ExtractAllData(dark));
        WriteF32Bin(Path.Combine(valDir, "ref_white.bin"), ExtractAllData(white));
        WriteF32Bin(Path.Combine(valDir, "ref_expected_avg16.bin"), cubeData);
        WriteMaskBin(Path.Combine(valDir, "ref_expected_mask.bin"), pipeResult.TissueMask!);
        // ToDo: No ref_expected_probs.bin , inference check will be skipped

        var manifest = MakeManifest(includePrep: true, includeValidation: true,
            refShape: [h, w, bandsIn], outShape: [h, w, outBands]);

        var result = ModelPackageValidator.Validate(_tempDir, manifest, session: null!);
        Assert.True(result.Passed, result.Summary);
        Assert.Equal(0f, result.PreprocessingMaxAbsDiff);
        Assert.True(result.MaskMatched);
    }

    [Fact]
    public void Preprocessing_fails_with_corrupted_golden_data()
    {
        const int h = 4, w = 4, bandsIn = 9;
        var prep = DefaultPrep();

        var raw = MakeBsqCube(h, w, bandsIn, seed: 42);
        var dark = MakeZeroBsqCube(h, w, bandsIn);
        var white = MakeOnesBsqCube(h, w, bandsIn);

        var pipeResult = PreprocessingService.Run(raw, dark, white, prep);
        var outBands = pipeResult.Cube.Bands;
        var cubeData = pipeResult.Cube.ExtractPatch(0, 0, w, h);

        // Save golden files but corrupt the expected output
        var valDir = Path.Combine(_tempDir, "validation");
        Directory.CreateDirectory(valDir);
        WriteF32Bin(Path.Combine(valDir, "ref_raw.bin"), ExtractAllData(raw));
        WriteF32Bin(Path.Combine(valDir, "ref_dark.bin"), ExtractAllData(dark));
        WriteF32Bin(Path.Combine(valDir, "ref_white.bin"), ExtractAllData(white));

        // Corrupt: add 1.0 to every value
        var corrupted = new float[cubeData.Length];
        for (var i = 0; i < corrupted.Length; i++)
            corrupted[i] = cubeData[i] + 1.0f;
        WriteF32Bin(Path.Combine(valDir, "ref_expected_avg16.bin"), corrupted);
        WriteMaskBin(Path.Combine(valDir, "ref_expected_mask.bin"), pipeResult.TissueMask!);

        var manifest = MakeManifest(includePrep: true, includeValidation: true,
            refShape: [h, w, bandsIn], outShape: [h, w, outBands]);

        var result = ModelPackageValidator.Validate(_tempDir, manifest, session: null!);
        Assert.False(result.Passed);
        Assert.Contains("FAILED", result.Summary);
        Assert.True(result.PreprocessingMaxAbsDiff > 0.5f);
    }


    // -- Helpers -- //

    private static readonly List<string> DefaultSteps =
        ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"];

    private static PreprocessingConfig DefaultPrep() => new()
    {
        CalibrationEpsilon = 1e-8f,
        ClipMin = 0f,
        ClipMax = 1f,
        NeighborAverageWindow = 3,
        BandReduceOutBands = 3, // 9 bands / window 3 = 3 after avg3; use uneven
        BandReduceStrategy = "uneven",
        TissueMaskMethod = "mean_std_percentile",
        TissueMaskQMean = 0.5f,
        TissueMaskQStd = 0.4f,
        TissueMaskMinObjectSize = 1,
        TissueMaskMinHoleSize = 1,
    };

    private static ModelManifest MakeManifest(
        bool includePrep,
        bool includeValidation,
        int[]? refShape = null,
        int[]? outShape = null)
    {
        var manifest = new ModelManifest
        {
            SchemaVersion = "1.1",
            InputSpec = new InputSpec
            {
                InputRank = 5,
                TensorLayout = "NCDHW",
                ExpectedBands = 3,
                SpatialPatchSize = [4, 4],
            },
        };
        if (includePrep)
            manifest.Pipeline = new PipelineConfig
            {
                Preprocessing = new PreprocessingPipeline { Params = DefaultPrep() }
            };
        if (includeValidation)
            manifest.Validation = new ValidationSpec
            {
                RefCubeShape = refShape != null ? [.. refShape] : [4, 4, 9],
                ExpectedOutputShape = outShape != null ? [.. outShape] : [4, 4, 3],
                PreprocessingTolerance = 1e-5f,
                InferenceTolerance = 1e-4f,
            };
        return manifest;
    }

    private static HsiCube MakeBsqCube(int h, int w, int b, int seed)
    {
        var rng = new Random(seed);
        var data = new float[b * h * w];
        for (var i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 0.5);
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }

    private static HsiCube MakeZeroBsqCube(int h, int w, int b)
    {
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, new float[b * h * w]);
    }

    private static HsiCube MakeOnesBsqCube(int h, int w, int b)
    {
        var data = new float[b * h * w];
        Array.Fill(data, 1f);
        var header = new HsiHeader { Lines = h, Samples = w, Bands = b, Interleave = "bsq" };
        return new HsiCube(header, data);
    }

    private static float[] ExtractAllData(HsiCube cube) =>
        cube.ExtractPatch(0, 0, cube.Samples, cube.Lines);

    private static void WriteF32Bin(string path, float[] data)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static void WriteMaskBin(string path, bool[] mask)
    {
        var bytes = new byte[mask.Length];
        for (var i = 0; i < mask.Length; i++)
            bytes[i] = mask[i] ? (byte)1 : (byte)0;
        File.WriteAllBytes(path, bytes);
    }
}
*/