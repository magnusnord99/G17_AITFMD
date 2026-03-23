using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public class OnnxClassifier : IClassifier, IDisposable
{
    private ModelPackage? _package;
    public ModelManifest? Manifest => _package?.Manifest;

    public void SetModel(ModelPackage package)
    {
        _package?.Dispose();
        _package = package;
    }
    
    public Task<ClassificationResult> ClassifyImageAsync(HsiCube cube)
    {
        if (_package == null)
            throw new InvalidOperationException("No model loaded");

        var spec = _package.Manifest.InputSpec;
        var patchW = spec.SpatialPatchSize[1];
        var patchH = spec.SpatialPatchSize[0];

        if (cube.Bands != spec.ExpectedBands)
            throw new InvalidOperationException(
                $"Model expects {spec.ExpectedBands} bands, image has {cube.Bands}");

        var predictions = new List<PatchPrediction>();

        for (var y = 0; y < cube.Lines; y += patchH)
        {
            var startY = Math.Min(y, cube.Lines - patchH);
            for (var x = 0; x < cube.Samples; x += patchW)
            {
                var startX = Math.Min(x, cube.Samples - patchW);

                var patchData = cube.ExtractPatch(startX, startY, patchW, patchH);
                var probs = ClassifyPatch(patchData);

                predictions.Add(new PatchPrediction
                {
                    X = startX,
                    Y = startY,
                    Probabilities = probs,
                });
            }
        }

        var tilesX = cube.Samples / patchW;
        var tilesY = cube.Lines / patchH;

        return Task.FromResult(new ClassificationResult
        {
            Predictions = predictions,
            ImageWidth = cube.Samples,
            ImageHeight = cube.Lines,
            PatchW = patchW,
            PatchH = patchH,
            Classes = _package.Manifest.OutputSpec.Classes,
            ModelName = _package.Manifest.Metadata.Name,
            TotalPossible = tilesX * tilesY,
            Evaluated = predictions.Count,
            Skipped = 0,
        });
    }

    private float[] ClassifyPatch(float[] patchData)
    {
        var spec = _package!.Manifest.InputSpec;

        var tensor = new DenseTensor<float>(
            patchData,
            new[] { 1, spec.ExpectedBands, spec.SpatialPatchSize[0], spec.SpatialPatchSize[1] }
        );

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", tensor)
        };

        using var results = _package.Session.Run(inputs);
        return results[0].AsEnumerable<float>().ToArray();
    }
    
    public void Dispose()
    {
        _package?.Dispose();
        _package = null;
        GC.SuppressFinalize(this);
    }
}
