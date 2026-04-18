using System;

namespace SpectralAssist.Services.Export;

/// <summary>Input payload for <see cref="PdfReportService"/>.</summary>
public sealed class PdfReportDocument
{
    public required DateTimeOffset InferenceCompletedAt { get; init; }
    public required DateTimeOffset ExportedAt { get; init; }
    public required string ManifestDisplayName { get; init; }
    public required string ModelNameFromResult { get; init; }
    public required string AccuracyDisplay { get; init; }
    public required string ReportSummaryText { get; init; }

    public required byte[] SyntheticRgbPng { get; init; }
    public required byte[] Overlay0Png { get; init; }
    public required byte[] Overlay50Png { get; init; }
    public required byte[] Overlay80Png { get; init; }
}
