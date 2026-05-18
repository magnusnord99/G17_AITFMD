using System;
using System.Collections.Generic;

namespace SpectralAssist.Services.Export;

/// <summary>
/// Input payload for <see cref="PdfReportExporter"/>.
/// </summary>
public sealed class PdfReportDocument
{
    // -- Identity --
    public required string ImageRelPath { get; init; }
    public required DateTimeOffset InferenceCompletedAt { get; init; }
    public required DateTimeOffset ExportedAt { get; init; }
    public required string ExecutionProvider { get; init; }

    // -- Model --
    public required string ModelDisplayName { get; init; }
    public required string Architecture { get; init; }
    public required string Task { get; init; }
    public required string ParameterCountDisplay { get; init; }
    public required IReadOnlyList<MetricRow> TestMetrics { get; init; }

    // -- Pipeline --
    public required IReadOnlyList<string> PreprocessingSteps { get; init; }
    public required string SpectralReducerDisplay { get; init; }
    public required string PatchSizeDisplay { get; init; }
    public required string StrideDisplay { get; init; }

    // -- Run statistics --
    public required int TotalPatches { get; init; }
    public required int EvaluatedPatches { get; init; }
    public required int SkippedPatches { get; init; }
    public required IReadOnlyList<ClassStatRow> ClassStatistics { get; init; }

    // -- Images --
    public required byte[] SyntheticRgbPng { get; init; }
    public required byte[] Overlay0Png { get; init; }
    public required byte[] Overlay50Png { get; init; }
    public required byte[] Overlay80Png { get; init; }

    public required float OverlayOpacity { get; init; }
    public required float Overlay1Threshold { get; init; }
    public required float Overlay2Threshold { get; init; }

    // -- Notes --
    public string? ImageNotes { get; init; }

    public sealed record MetricRow(string Label, double? Value);
    public sealed record ClassStatRow(string ClassName, int PercentWinner, double PercentAbove50, double PercentAbove80);
}
