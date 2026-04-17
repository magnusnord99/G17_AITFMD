using System;
using System.Linq;

namespace SpectralAssist.Models;

public class PatchPrediction
{
    public int X { get; init; }
    public int Y { get; init; }
    public float[] Probabilities { get; init; } = [];
    public float[] Logits { get; init; } = [];

    public int PredictedClass =>
        Array.IndexOf(Probabilities, Probabilities.Max());

    public float Confidence => Probabilities.Max();
}