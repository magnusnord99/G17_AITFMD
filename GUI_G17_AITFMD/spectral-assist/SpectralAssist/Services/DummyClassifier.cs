using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public class DummyClassifier : IClassifier
{
    private readonly int _patchW;
    private readonly int _patchH;
    private readonly int _stride;

    private DummyClassifier(int patchW, int patchH, int stride)
    {
        _patchW = patchW;
        _patchH = patchH;
        _stride = stride;
    }

    /// <summary>
    /// Random data for UI testing when no Onnx-model or Python is available.
    /// </summary>
    public static IClassifier Random(int patchW = 32, int patchH = 32, int stride = 32)
    {
        return new DummyClassifier(patchW, patchH, stride);
    }

    /// <summary>
    /// Load from a saved prediction.json file.
    /// </summary>
    public static IClassifier FromJson(string jsonPath)
    {
        return new JsonFileClassifier(jsonPath);
    }
    
    
    public Task<ClassificationResult> ClassifyImageAsync(HsiCube cube)
    {
        var predictions = new List<PatchPrediction>();

        for (var y = 0; y < cube.Lines; y += _stride)
        {
            var startY = Math.Min(y, cube.Lines - _patchH);
            for (var x = 0; x < cube.Samples; x += _stride)
            {
                var startX = Math.Min(x, cube.Samples - _patchW);
                var rng = new Random((startY * cube.Samples + startX) * 31 + 7);
                var tumor = (float)rng.NextDouble();

                predictions.Add(new PatchPrediction
                {
                    X = startX,
                    Y = startY,
                    Probabilities = [1f - tumor, tumor],
                });
            }
        }

        return Task.FromResult(new ClassificationResult
        {
            Predictions = predictions,
            ImageWidth = cube.Samples,
            ImageHeight = cube.Lines,
            PatchW = _patchW,
            PatchH = _patchH,
            Classes = ["non-tumor", "tumor"],
            ModelName = $"Fake Random (stride={_stride})",
            TotalPossible = predictions.Count,
            Evaluated = predictions.Count,
            Skipped = 0,
        });
    }

    /// <summary>
    /// Loads a prediction.json and parses it as a ClassificationResult.
    /// </summary>
    private class JsonFileClassifier(string jsonPath) : IClassifier
    {
        public Task<ClassificationResult> ClassifyImageAsync(HsiCube cube)
        {
            var json = File.ReadAllText(jsonPath);
            var result = PythonClassifier.ParsePredictionJson(json, cube);
            return Task.FromResult(result);
        }
    }
}