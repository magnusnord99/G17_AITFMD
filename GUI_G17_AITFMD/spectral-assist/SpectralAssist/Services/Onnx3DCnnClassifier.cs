using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// ONNX Runtime for 3D-CNN eksportert med shape <c>[1, 1, C, H, W]</c> (NCDHW), som i
/// <c>pytorch_backend._patches_to_tensor</c>: (N,H,W,C) → (N,1,C,H,W).
/// </summary>
public class Onnx3DCnnClassifier : IClassifier, IDisposable
{
    private ModelPackage? _package;

    public ModelManifest? Manifest => _package?.Manifest;

    public void SetModel(ModelPackage package)
    {
        _package?.Dispose();
        _package = package;
        ValidateManifest(_package.Manifest);
    }

    public Task<ClassificationResult> ClassifyImageAsync(HsiCube cube)
    {
        if (_package == null)
            throw new InvalidOperationException("No model loaded");

        var spec = _package.Manifest.InputSpec;
        var patchH = spec.SpatialPatchSize[0];
        var patchW = spec.SpatialPatchSize[1];
        var bands = spec.SpectralBands ?? spec.ExpectedBands;

        if (cube.Bands != bands)
            throw new InvalidOperationException(
                $"Model expects {bands} spectral bands, cube has {cube.Bands}.");

        var predictions = new List<PatchPrediction>();

        for (var y = 0; y < cube.Lines; y += patchH)
        {
            var startY = Math.Min(y, cube.Lines - patchH);
            for (var x = 0; x < cube.Samples; x += patchW)
            {
                var startX = Math.Min(x, cube.Samples - patchW);

                var patchData = cube.ExtractPatch(startX, startY, patchW, patchH);
                var probs = ClassifyPatchLogitsToProbabilities(patchData, bands, patchH, patchW);

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

    private float[] ClassifyPatchLogitsToProbabilities(float[] patchBsq, int bands, int patchH, int patchW)
    {
        var tensor = new DenseTensor<float>(
            patchBsq,
            new[] { 1, 1, bands, patchH, patchW });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", tensor),
        };

        using var results = _package!.Session.Run(inputs);
        var logits = results[0].AsEnumerable<float>().ToArray();
        return Softmax(logits);
    }

    private static float[] Softmax(float[] logits)
    {
        if (logits.Length == 0)
            return logits;
        var max = logits.Max();
        var exp = new float[logits.Length];
        float sum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            sum += exp[i];
        }

        if (sum <= 0f)
            return logits;
        for (var i = 0; i < logits.Length; i++)
            exp[i] /= sum;
        return exp;
    }

    private static void ValidateManifest(ModelManifest manifest)
    {
        var spec = manifest.InputSpec;
        if (spec.SpatialPatchSize.Count < 2)
            throw new InvalidOperationException("manifest.input_spec.spatial_patch_size must have length >= 2.");

        var rank = spec.InputRank ?? (spec.InputShape?.Count == 5 ? 5 : 4);
        if (rank != 5)
            throw new InvalidOperationException(
                "Onnx3DCnnClassifier requires input_rank 5 or input_shape with 5 dimensions (NCDHW). " +
                "Use OnnxClassifier for 4D NCHW models.");

        if (spec.InputShape is { Count: 5 } shape)
        {
            if (shape[2] != (spec.SpectralBands ?? spec.ExpectedBands))
                throw new InvalidOperationException(
                    "manifest input_shape[2] must match spectral band count.");
        }
    }

    public void Dispose()
    {
        _package?.Dispose();
        _package = null;
        GC.SuppressFinalize(this);
    }
}
