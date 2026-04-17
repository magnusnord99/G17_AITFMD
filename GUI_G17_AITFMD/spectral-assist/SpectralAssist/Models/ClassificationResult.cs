using System.Collections.Generic;

namespace SpectralAssist.Models;

public class ClassificationResult
{
    public List<PatchPrediction> Predictions { get; init; } = [];

    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public int PatchW { get; init; }
    public int PatchH { get; init; }

    public List<string> Classes { get; init; } = [];
    public string ModelName { get; init; } = "";

    public int TotalPossible { get; init; }
    public int Evaluated { get; init; }
    public int Skipped { get; init; }

    // ToDo: Remove this before deployment
    public string ExecutionProvider { get; init; } = "Unknown";
}