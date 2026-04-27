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
    /// <param name="report">The classification report containing model output and metadata</param>
    /// <param name="syntheticRgb">A synthetic RGB bitmap generated from the cube.</param>
    /// <param name="heatmap">A sequence of intensity values (0-1) used to generate the heatmap.</param>
    /// <param name="options">Selected export settings controlling content.</param>
    /// <param name="output">The destination stream to write the generated PDF to.</param>
    public static void Export(
        ClassificationReport report,
        Bitmap syntheticRgb,
        float[] heatmap,
        ExportOptions options,
        Stream output)
    {
        var document = BuildDocument(report, syntheticRgb, heatmap, options);
        Write(output, document);
    }

    // Build and Write =========================================================================
    
    private static PdfReportDocument BuildDocument(ClassificationReport report, Bitmap syntheticRgb, float[] heatmap,
        ExportOptions options)
    {
        var imageWidth = report.ImageWidth;
        var imageHeight = report.ImageHeight;
        var colorMap = ColorMaps.All.GetValueOrDefault(options.ColorMapName, ColorMaps.All.Values.First());
        var accuracy = report.ModelTraining.Metrics.Accuracy;
        var accDisplay = accuracy?.ToString("P1") ?? "—";
        
        Bitmap? c0 = null, c1 = null, c2 = null;
        try
        {
            using var ol0 = HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, 0f);
            using var ol1 =
                HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, options.Overlay1Threshold);
            using var ol2 =
                HeatmapRenderer.RenderHeatmap(heatmap, imageWidth, imageHeight, colorMap, options.Overlay2Threshold);

            c0 = RgbOverlayComposer.Compose(syntheticRgb, ol0, options.OverlayOpacity);
            c1 = RgbOverlayComposer.Compose(syntheticRgb, ol1, options.OverlayOpacity);
            c2 = RgbOverlayComposer.Compose(syntheticRgb, ol2, options.OverlayOpacity);

            return new PdfReportDocument
            {
                InferenceCompletedAt = new DateTimeOffset(report.CompletedAt, TimeSpan.Zero),
                ExportedAt = DateTimeOffset.Now,
                ModelDisplayName = report.ModelDisplayName,
                ModelName = report.ModelMetadata.Name,
                AccuracyDisplay = accDisplay,
                ReportSummaryText = ClassificationReportMetrics.BuildReportSummaryText(report),
                SyntheticRgbPng = EncodeForPdf(syntheticRgb),
                Overlay0Png = EncodeForPdf(c0),
                Overlay50Png = EncodeForPdf(c1),
                Overlay80Png = EncodeForPdf(c2),
                Overlay1Threshold = options.Overlay1Threshold,
                Overlay2Threshold = options.Overlay2Threshold,
                OverlayOpacity = options.OverlayOpacity,
                RunNotes = string.IsNullOrWhiteSpace(report.Notes) ? null : report.Notes,
            };
        }
        finally
        {
            c0?.Dispose();
            c1?.Dispose();
            c2?.Dispose();
        }
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
                page.Content().PaddingTop(20).Column(col =>
                {
                    col.Spacing(18);
                    col.Item().Element(c => MetadataSection(c, doc));
                    col.Item().Element(c => SummarySection(c, doc));
                    if (!string.IsNullOrWhiteSpace(doc.RunNotes))
                        col.Item().Element(c => NotesSection(c, doc));
                    col.Item().Element(c => ImagesSection(c, doc));
                });
                page.Footer().Element(Footer);
            });
        }).GeneratePdf(output);
    }
    
    // Header =========================================================================
    
    private static void Header(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item()
                .Background(ColorPrimary)
                .Padding(20)
                .Row(row =>
                {
                    row.RelativeItem().Column(c =>
                    {
                        c.Item().Text("SpectralAssist")
                            .FontSize(22).Bold().FontColor("#FFFFFF");
                        c.Item().Text("Hyperspectral Image Classification Report")
                            .FontSize(10).FontColor("#A0C4E8");
                    });
                    row.ConstantItem(130).AlignRight().AlignMiddle().Column(c =>
                    {
                        c.Item().AlignRight().Text(doc.ExportedAt.LocalDateTime.ToString("yyyy-MM-dd"))
                            .FontSize(9).FontColor("#A0C4E8");
                        c.Item().AlignRight().Text(doc.ExportedAt.LocalDateTime.ToString("HH:mm:ss"))
                            .FontSize(9).FontColor("#A0C4E8");
                    });
                });

            col.Item().Height(3).Background(ColorAccent);
        });
    }

    // Metadata =========================================================================
    
    private static void MetadataSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Report Details").SemiBold().FontSize(11);
            col.Item().PaddingTop(6).Border(1).BorderColor(ColorBorder).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.ConstantColumn(170);
                    cols.RelativeColumn();
                });

                MetaRow(table, "Package", doc.ModelDisplayName, shaded: false);
                MetaRow(table, "Model", doc.ModelName, shaded: true);
                MetaRow(table, "Validation accuracy", doc.AccuracyDisplay, shaded: false);
                MetaRow(table, "Inference completed",
                    doc.InferenceCompletedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"), shaded: true);
                MetaRow(table, "Report exported",
                    doc.ExportedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"), shaded: false);
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

    
    // Summary =========================================================================
    
    private static void SummarySection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Run Summary").SemiBold().FontSize(11);
            col.Item().PaddingTop(6)
                .BorderLeft(3).BorderColor(ColorAccent)
                .Background(ColorSurface)
                .PaddingHorizontal(14).PaddingVertical(10)
                .Text(doc.ReportSummaryText.TrimEnd())
                .FontFamily("Courier New").FontSize(9).FontColor("#2D3748");
        });
    }

    // ── Notes ────────────────────────────────────────────────────────────────

    private static void NotesSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Notes").SemiBold().FontSize(11);
            col.Item().PaddingTop(6)
                .Border(1).BorderColor(ColorBorder)
                .Background(ColorSurface)
                .PaddingHorizontal(14).PaddingVertical(10)
                .Text(doc.RunNotes!.TrimEnd())
                .FontSize(10).FontColor("#2D3748");
        });
    }

    // ── Images ───────────────────────────────────────────────────────────────

    private static void ImagesSection(IContainer container, PdfReportDocument doc)
    {
        container.Column(col =>
        {
            col.Item().Text("Images").SemiBold().FontSize(11);
            col.Item().PaddingTop(6).Table(table =>
            {
                table.ColumnsDefinition(cols =>
                {
                    cols.RelativeColumn();
                    cols.ConstantColumn(10);
                    cols.RelativeColumn();
                });

                table.Cell().Row(1).Column(1).Element(c => Figure(c, "Synthetic RGB", doc.SyntheticRgbPng));
                table.Cell().Row(1).Column(2); // gap
                table.Cell().Row(1).Column(3).Element(c =>
                    Figure(c, $"Overlay — threshold 0 %, opacity {doc.OverlayOpacity:P0}", doc.Overlay0Png));

                table.Cell().Row(2).Column(1).PaddingTop(10).Element(c => Figure(c,
                    $"Overlay — threshold {doc.Overlay1Threshold:P0}, opacity {doc.OverlayOpacity:P0}",
                    doc.Overlay50Png));
                table.Cell().Row(2).Column(2); // gap
                table.Cell().Row(2).Column(3).PaddingTop(10).Element(c => Figure(c,
                    $"Overlay — threshold {doc.Overlay2Threshold:P0}, opacity {doc.OverlayOpacity:P0}",
                    doc.Overlay80Png));
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


    // Footer =========================================================================

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