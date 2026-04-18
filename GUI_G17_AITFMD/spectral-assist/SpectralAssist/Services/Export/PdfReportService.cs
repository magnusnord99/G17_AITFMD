using System.IO;
using QuestPDF.Fluent;
using QuestPDF.Infrastructure;

namespace SpectralAssist.Services.Export;

public sealed class PdfReportService
{
    private const string ColorPrimary  = "#1E3A5F";
    private const string ColorAccent   = "#3B82F6";
    private const string ColorSurface  = "#F7FAFC";
    private const string ColorBorder   = "#CBD5E0";
    private const string ColorTextMuted = "#4A5568";

    public PdfReportService()
    {
        QuestPDF.Settings.License = LicenseType.Community;
    }

    public void Write(Stream output, PdfReportDocument doc)
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
                    col.Item().Element(c => ImagesSection(c, doc));
                });
                page.Footer().Element(Footer);
            });
        }).GeneratePdf(output);
    }

    // ── Header ──────────────────────────────────────────────────────────────

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

    // ── Metadata ─────────────────────────────────────────────────────────────

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

                MetaRow(table, "Package",              doc.ManifestDisplayName, shaded: false);
                MetaRow(table, "Model",                doc.ModelNameFromResult,  shaded: true);
                MetaRow(table, "Validation accuracy",  doc.AccuracyDisplay,      shaded: false);
                MetaRow(table, "Inference completed",
                    doc.InferenceCompletedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"), shaded: true);
                MetaRow(table, "Report exported",
                    doc.ExportedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"),           shaded: false);
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

    // ── Summary ──────────────────────────────────────────────────────────────

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
                table.Cell().Row(1).Column(3).Element(c => Figure(c, "Overlay — threshold 0 %, opacity 50 %", doc.Overlay0Png));

                table.Cell().Row(2).Column(1).PaddingTop(10).Element(c => Figure(c, "Overlay — threshold 50 %, opacity 50 %", doc.Overlay50Png));
                table.Cell().Row(2).Column(2); // gap
                table.Cell().Row(2).Column(3).PaddingTop(10).Element(c => Figure(c, "Overlay — threshold 80 %, opacity 50 %", doc.Overlay80Png));
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

    // ── Footer ───────────────────────────────────────────────────────────────

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
