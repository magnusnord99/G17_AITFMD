using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace SpectralAssist.Models;

public class ClassificationReport
{
    // Identity
    public string RunId { get; init; } = string.Empty;
    public string ImageId { get; init; } = string.Empty;
    public string ModelId { get; init; } = string.Empty;
    public string ModelName { get; init; } = string.Empty;
    public DateTime DatePerformed { get; init; }

    // Metadata
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public int PatchH { get; init; }
    public int PatchW { get; init; }
    public int StrideH { get; init; }
    public int StrideW { get; init; }
    
    // Data
    public int TotalPatches { get; init; }
    public int EvaluatedPatches { get; init; }
    public int SkippedPatches { get; init; }
    public string ExecutionProvider { get; init; } = string.Empty;
    
    public List<string> Classes { get; init; } = [];
    public List<ClassStatistics> Statistics { get; set; } = [];
    public List<PatchPrediction> Predictions { get; init; } = [];
}

public class PatchPrediction
{
    public int X { get; init; }
    public int Y { get; init; }
    public float[] Probabilities { get; init; } = [];

    [JsonIgnore] public float[] Logits { get; init; } = [];

    [JsonIgnore]
    public int PredictedClass =>
        Array.IndexOf(Probabilities, Probabilities.Max());

    [JsonIgnore] public float Confidence => Probabilities.Max();
}

public class ClassStatistics
{
    public int ClassIndex { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public int PercentWinner { get; init; }
    public double PercentAbove50 { get; init; }
    public double PercentAbove80 { get; init; }
}