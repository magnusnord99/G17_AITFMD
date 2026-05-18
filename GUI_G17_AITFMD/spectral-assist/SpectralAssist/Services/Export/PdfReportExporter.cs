using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Avalonia.Media.Imaging;
using QuestPDF.Fluent;
using QuestPDF.Infrastructure;
using SpectralAssist.Models;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.Services.Export;

public static class PdfReportExporter
{
    private const string ColorPrimary = "#1E3A5F";
    private const string ColorAccent = "#3B82F6";
    private const string ColorSurface = "#F7FAFC";
    private const string ColorBorder = "#CBD5E0";
    private const string ColorTextMuted = "#4A5568";

    static PdfReportExporter()
    {
        QuestPDF.Settings.License = LicenseType.Community;
    }

    /// <summary>
    /// Generates a complete PDF report for the given inference result and
    /// writes it directly to the provided <paramref name="output"/> stream.
    /// </summary>
    public static void Export(
        ClassificationReport report,
        Bitmap syntheticRgb,
        float[] heatmap,
        ExportOptions options,
        Stream output,
        string imageRelPath,
        string? imageNotes = null)
    {
        var document = BuildDocument(report, syntheticRgb, heatmap, options, imageRelPath, imageNotes);
        Write(output, document);
    }
    
    /// <summary>
    /// Orchestrates the full PDF generation pipeline, combining report data,
    /// images, heatmaps, and export options into a complete PdfReportDocument.
    /// </summary>
    private static PdfReportDocument BuildDocument(ClassificationReport report, Bitmap syntheticRgb, float[] heatmap,
        ExportOptions options, string imageRelPath, string? imageNotes)
    {
        var imageWidth = report.ImageWidth;
        var imageHeight = report.ImageHeight;
        var colorMap = ColorMaps.All.GetValueOrDefault(options.ColorMapName, ColorMaps.All.Values.First());

        Bitmap? c0 = null, c1 = null, c2 = null;
        try
        {
            using var ol0 = HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, 0f);
            using var ol1 = HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, options.Overlay1Threshold);
            using var ol2 = HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, options.Overlay2Threshold);

            c0 = RgbOverlayComposer.Compose(syntheticRgb, ol0, options.OverlayOpacity);
            c1 = RgbOverlayComposer.Compose(syntheticRgb, ol1, options.OverlayOpacity);
            c2 = RgbOverlayComposer.Compose(syntheticRgb, ol2, options.OverlayOpacity);

            return new PdfReportDocument
            {
                ImageRelPath = string.IsNullOrWhiteSpace(imageRelPath) ? "—" : imageRelPath,
                InferenceCompletedAt = new DateTimeOffset(report.CompletedAt, TimeSpan.Zero),
                ExportedAt = DateTimeOffset.Now,
                ExecutionProvider = string.IsNullOrWhiteSpace(report.ExecutionProvider) ? "—" : report.ExecutionProvider,

                ModelDisplayName = report.ModelDisplayName,
                Architecture = string.IsNullOrWhiteSpace(report.ModelSummary.Architecture) ? "—" : report.ModelSummary.Architecture,
                Task = string.IsNullOrWhiteSpace(report.ModelSummary.Task) ? "—" : report.ModelSummary.Task,
                ParameterCountDisplay = FormatParameterCount(report.ModelSummary.TotalParameters),
                TestMetrics = BuildMetricRows(report.ModelTraining),

                PreprocessingSteps = report.PreprocessingSteps.Count > 0 ? report.PreprocessingSteps : ["—"],
                SpectralReducerDisplay = FormatSpectralReducer(report.SpectralReducer),
                PatchSizeDisplay = $"{report.PatchW} × {report.PatchH}",
                StrideDisplay = $"{report.StrideW} × {report.StrideH}",

                TotalPatches = report.TotalPatches,
                EvaluatedPatches = report.EvaluatedPatches,
                SkippedPatches = report.SkippedPatches,
                ClassStatistics = report.Statistics
                    .Select(s => new PdfReportDocument.ClassStatRow(
                        s.ClassName, s.PercentWinner, s.PercentAbove50, s.PercentAbove80))
                    .ToList(),

                SyntheticRgbPng = EncodeForPdf(syntheticRgb),
                Overlay0Png = EncodeForPdf(c0),
                Overlay50Png = EncodeForPdf(c1),
                Overlay80Png = EncodeForPdf(c2),
                Overlay1Threshold = options.Overlay1Threshold,
                Overlay2Threshold = options.Overlay2Threshold,
                OverlayOpacity = options.OverlayOpacity,
                ImageNotes = string.IsNullOrWhiteSpace(imageNotes) ? null : imageNotes,
            };
        }
        finally
        {
            c0?.Dispose();
            c1?.Dispose();
            c2?.Dispose();
        }
    }

    private static IReadOnlyList<PdfReportDocument.MetricRow> BuildMetricRows(TrainingInfo training)
    {
        var testMetrics = training.TestMetrics;
        {
            return
            [
                new PdfReportDocument.MetricRow("Accuracy", testMetrics.Accuracy),
                new PdfReportDocument.MetricRow("Precision", testMetrics.Precision),
                new PdfReportDocument.MetricRow("Recall", testMetrics.Recall),
                new PdfReportDocument.MetricRow("F1", testMetrics.F1),
                new PdfReportDocument.MetricRow("AUC-ROC", testMetrics.AucRoc),
            ];
        }
    }

    private static string FormatSpectralReducer(SpectralReducerInfo reducerInfo)
    {
        if (string.IsNullOrWhiteSpace(reducerInfo.Method)) 
            return "—";
        
        if (reducerInfo is { InputBands: { } inB, OutputBands: { } outB }) 
            return $"{reducerInfo.Method} ({inB} → {outB} bands)";
        
        return reducerInfo.Method;
    }

    private static string FormatParameterCount(long? totalParamters)
    {
        if (totalParamters is not { } value) 
            return "—";
        
        return value switch
        {
            >= 1_000_000 => $"{value / 1_000_000.0:F2}M",
            >= 1_000 => $"{value / 1_000.0:F1}k",
            _ => value.ToString()
        };
    }

    private static byte[] EncodeForPdf(Bitmap original)
    {
        var scaled = BitmapExportHelper.MaybeDownscale(original);
        try
        {
            return BitmapExportHelper.ToPngBytes(scaled);
        }
        finally
        {
            if (!ReferenceEquals(scaled, original))
                scaled.Dispose();
        }
    }
    
    /// <summary>
    /// Generates a PDF from the provided <see cref="PdfReportDocument"/> and writes
    /// the resulting PDF bytes to the specified output stream.
    /// </summary>
    private static void Write(Stream output, PdfReportDocument doc)
    {
        Document.Create(container =>
        {
            container.Page(page =>
            {
                page.MarginHorizontal(40);
                page.MarginVertical(32);
                page.DefaultTextStyle(x => x.FontSize(10).FontFamily("Helvetica").FontColor("#1A202C"));

                page.Header().Element(c => Header(c, doc));
                page.Content().PaddingTop(16).Column(col =>
                {
                    col.Spacing(16);

                    // -- First Page -- //
                    col.Item().Element(c => ImagesSection(c, doc));
                    if (!string.IsNullOrWhiteSpace(doc.ImageNotes))
                        col.Item().Element(c => NotesSection(c, doc));
                    
                    // -- Second Page -- //
                    col.Item().PageBreak();
                    col.Item().Element(c => RunStatisticsSection(c, doc));
                    col.Item().Element(c => ModelSection(c, doc));
                    col.Item().Element(c => PipelineSection(c, doc));
                });
                page.Footer().Element(Footer);
            });
        }).GeneratePdf(output);
    }

    
    private static void Header(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item()
                .Background(ColorPrimary)
                .Padding(18)
                .Row(row =>
                {
                    row.RelativeItem().Column(c =>
                    {
                        c.Item().Text("SpectralAssist")
                            .FontSize(20).Bold().FontColor("#FFFFFF");
                        c.Item().Text(doc.ImageRelPath)
                            .FontSize(11).FontColor("#FFFFFF");
                        c.Item().PaddingTop(2).Text("Hyperspectral classification report")
                            .FontSize(9).FontColor("#A0C4E8");
                    });
                    row.ConstantItem(160).AlignRight().AlignMiddle().Column(c =>
                    {
                        c.Item().AlignRight().Text("Inference completed")
                            .FontSize(8).FontColor("#A0C4E8");
                        c.Item().AlignRight().Text(doc.InferenceCompletedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm"))
                            .FontSize(10).FontColor("#FFFFFF").SemiBold();
                        c.Item().PaddingTop(4).AlignRight().Text("Exported")
                            .FontSize(8).FontColor("#A0C4E8");
                        c.Item().AlignRight().Text(doc.ExportedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm"))
                            .FontSize(9).FontColor("#A0C4E8");
                    });
                });

            col.Item().Height(3).Background(ColorAccent);
        });
    }
    
    private static void ImagesSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Findings").SemiBold().FontSize(11);
            col.Item().PaddingTop(6).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.RelativeColumn();
                    cols.ConstantColumn(10);
                    cols.RelativeColumn();
                });

                table.Cell().Row(1).Column(1).Element(c => Figure(c, "Synthetic RGB", doc.SyntheticRgbPng));
                table.Cell().Row(1).Column(2);
                table.Cell().Row(1).Column(3).Element(c =>
                    Figure(c, $"Overlay — all patches (opacity {doc.OverlayOpacity:P0})", doc.Overlay0Png));

                table.Cell().Row(2).Column(1).PaddingTop(10).Element(c => Figure(c,
                    $"Overlay — threshold {doc.Overlay1Threshold:P0}", doc.Overlay50Png));
                table.Cell().Row(2).Column(2).PaddingTop(10);
                table.Cell().Row(2).Column(3).PaddingTop(10).Element(c => Figure(c,
                    $"Overlay — threshold {doc.Overlay2Threshold:P0}", doc.Overlay80Png));
            });
        });
    }

    private static void Figure(IContainer container, string caption, byte[] png)
    {
        container.Column(col =>
        {
            col.Item().Border(1).BorderColor(ColorBorder).Image(png).FitWidth();
            col.Item().PaddingTop(4).AlignCenter()
                .Text(caption).Italic().FontSize(8).FontColor(ColorTextMuted);
        });
    }
    
    private static void NotesSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Notes").SemiBold().FontSize(11);
            col.Item().PaddingTop(6)
                .Border(1).BorderColor(ColorBorder)
                .Background(ColorSurface)
                .PaddingHorizontal(14).PaddingVertical(10)
                .Text(doc.ImageNotes!.TrimEnd())
                .FontSize(10).FontColor("#2D3748");
        });
    }
    
    private static void ModelSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Model").SemiBold().FontSize(11);

            col.Item().PaddingTop(6).Border(1).BorderColor(ColorBorder).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.ConstantColumn(170);
                    cols.RelativeColumn();
                });

                MetaRow(table, "Package", doc.ModelDisplayName, shaded: false);
                MetaRow(table, "Architecture", doc.Architecture, shaded: true);
                MetaRow(table, "Task", doc.Task, shaded: false);
                MetaRow(table, "Parameters", doc.ParameterCountDisplay, shaded: true);
            });

            col.Item().PaddingTop(10).Text("Test metrics").FontSize(9).FontColor(ColorTextMuted);
            col.Item().PaddingTop(4).Border(1).BorderColor(ColorBorder).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    foreach (var _ in doc.TestMetrics) cols.RelativeColumn();
                });

                foreach (var m in doc.TestMetrics)
                {
                    table.Cell()
                        .Background(ColorSurface).BorderRight(1).BorderColor(ColorBorder)
                        .PaddingHorizontal(8).PaddingVertical(6)
                        .AlignCenter()
                        .Text(m.Label).FontSize(8).FontColor(ColorTextMuted);
                }
                foreach (var m in doc.TestMetrics)
                {
                    table.Cell()
                        .BorderRight(1).BorderColor(ColorBorder)
                        .PaddingHorizontal(8).PaddingVertical(8)
                        .AlignCenter()
                        .Text(m.Value is { } v ? v.ToString("P1") : "—").SemiBold().FontSize(11);
                }
            });
        });
    }

    private static void MetaRow(TableDescriptor table, string label, string value, bool shaded)
    {
        var bg = shaded ? ColorSurface : "#FFFFFF";

        table.Cell()
            .Background(bg).BorderBottom(1).BorderColor(ColorBorder)
            .PaddingHorizontal(12).PaddingVertical(7)
            .Text(label).FontColor(ColorTextMuted);

        table.Cell()
            .Background(bg).BorderBottom(1).BorderColor(ColorBorder)
            .PaddingHorizontal(12).PaddingVertical(7)
            .Text(value).SemiBold();
    }
    
    private static void PipelineSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Pipeline").SemiBold().FontSize(11);
            col.Item().PaddingTop(6).Border(1).BorderColor(ColorBorder).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.ConstantColumn(170);
                    cols.RelativeColumn();
                });

                MetaRow(table, "Preprocessing steps", string.Join(" → ", doc.PreprocessingSteps), shaded: false);
                MetaRow(table, "Spectral reducer", doc.SpectralReducerDisplay, shaded: true);
                MetaRow(table, "Patch size", doc.PatchSizeDisplay, shaded: false);
                MetaRow(table, "Stride", doc.StrideDisplay, shaded: true);
                MetaRow(table, "Execution provider", doc.ExecutionProvider, shaded: false);
            });
        });
    }
    
    private static void RunStatisticsSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Run Statistics").SemiBold().FontSize(11);

            col.Item().PaddingTop(6).Row(row =>
            {
                row.RelativeItem().Element(c => StatTile(c, "Total patches", doc.TotalPatches.ToString()));
                row.ConstantItem(10);
                row.RelativeItem().Element(c => StatTile(c, "Evaluated", doc.EvaluatedPatches.ToString()));
                row.ConstantItem(10);
                row.RelativeItem().Element(c => StatTile(c, "Skipped (background)", doc.SkippedPatches.ToString()));
            });

            if (doc.ClassStatistics.Count <= 0) return;
            
            col.Item().PaddingTop(10).Text("Per-class distribution").FontSize(9).FontColor(ColorTextMuted);
            col.Item().PaddingTop(4).Border(1).BorderColor(ColorBorder).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.RelativeColumn(2);
                    cols.RelativeColumn();
                    cols.RelativeColumn();
                    cols.RelativeColumn();
                });

                StatHeaderCell(table, "Class");
                StatHeaderCell(table, "Winner");
                StatHeaderCell(table, "P > 50 %");
                StatHeaderCell(table, "P > 80 %");

                var shaded = false;
                foreach (var s in doc.ClassStatistics)
                {
                    var bg = shaded ? ColorSurface : "#FFFFFF";
                    StatCell(table, s.ClassName, bg, bold: true, align: false);
                    StatCell(table, $"{s.PercentWinner} %", bg, bold: false, align: true);
                    StatCell(table, $"{s.PercentAbove50:F1} %", bg, bold: false, align: true);
                    StatCell(table, $"{s.PercentAbove80:F1} %", bg, bold: false, align: true);
                    shaded = !shaded;
                }
            });
        });
    }

    private static void StatTile(IContainer container, string label, string value)
    {
        container
            .Border(1).BorderColor(ColorBorder)
            .Background(ColorSurface)
            .PaddingVertical(12).PaddingHorizontal(14)
            .Column(c =>
            {
                c.Item().Text(label).FontSize(8).FontColor(ColorTextMuted);
                c.Item().PaddingTop(2).Text(value).FontSize(16).SemiBold().FontColor(ColorPrimary);
            });
    }

    private static void StatHeaderCell(TableDescriptor table, string label)
    {
        table.Cell()
            .Background(ColorPrimary).BorderBottom(1).BorderColor(ColorBorder)
            .PaddingHorizontal(8).PaddingVertical(5)
            .Text(label).FontColor("#FFFFFF").FontSize(8).SemiBold();
    }

    private static void StatCell(TableDescriptor table, string text, string bg, bool bold, bool align)
    {
        var cell = table.Cell()
            .Background(bg).BorderBottom(1).BorderColor(ColorBorder)
            .PaddingHorizontal(8).PaddingVertical(5);
        var t = align ? cell.AlignRight().Text(text) : cell.Text(text);
        if (bold) t.SemiBold();
    }
    
    private static void Footer(IContainer container)
    {
        container
            .BorderTop(1).BorderColor(ColorBorder)
            .PaddingTop(8)
            .Row(row =>
            {
                row.RelativeItem()
                    .Text("SpectralAssist — G17 AITFMD")
                    .FontSize(8).FontColor(ColorTextMuted);

                row.RelativeItem().AlignRight().Text(x =>
                {
                    x.Span("Page ").FontSize(8).FontColor(ColorTextMuted);
                    x.CurrentPageNumber().FontSize(8).FontColor(ColorTextMuted);
                    x.Span(" of ").FontSize(8).FontColor(ColorTextMuted);
                    x.TotalPages().FontSize(8).FontColor(ColorTextMuted);
                });
            });
    }
}
