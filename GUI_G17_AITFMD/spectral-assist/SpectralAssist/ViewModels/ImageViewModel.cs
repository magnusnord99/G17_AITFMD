using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Export;
using SpectralAssist.Services.Library;
using SpectralAssist.Services.Rendering;
using SpectralAssist.ViewModels.Components;
using SpectralAssist.Views;

namespace SpectralAssist.ViewModels;

public enum LoadingState
{
    Idle,
    Loading,
    Ready,
    Error
}

/// <summary>
/// Coordinator ViewModel for the image analysis view.
/// Orchestrates three independent stages: Load → Preprocess → Infer.
/// Each stage is handled by its own service:
/// <list>
/// <item><see cref="ImageLoadingService"/> loads and calibrates the HSI cube</item>
/// <item><see cref="PreprocessingService"/> for manifest-driven preprocessing (static class)</item>
/// <item><see cref="InferenceService"/> for ONNX model inference</item>
/// </list>
/// Overlay state is managed by <see cref="OverlayViewModel"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase, IDisposable
{
    // Single-file mode constructor: FilePicker and Drag&Drop without persistence
    public ImageViewModel(
        string hdrPath, InferenceService inferenceService, PdfReportService? pdfReportService = null)
    {
        _hdrPath = hdrPath;
        _inferenceService = inferenceService;
        _pdfReportService = pdfReportService;
        _ = LoadAsync();
    }

    // Library mode constructor: library context aware with persistence
    public ImageViewModel(string hdrPath, InferenceService inferenceService,
        string? imageId, LibraryManager? libraryManager, PdfReportService? pdfReportService = null)
        : this(hdrPath, inferenceService, pdfReportService)
    {
        _imageId = imageId;
        _libraryManager = libraryManager;

        if (libraryManager != null && imageId != null)
        {
            var imageNode = libraryManager.FindImage(imageId);
            _imageNode = imageNode;
            foreach (var r in imageNode?.Runs ?? Enumerable.Empty<RunSummary>())
                Reports.Add(r);
        }
    }

    private readonly ImageNode? _imageNode;
    private readonly string? _imageId;
    private readonly LibraryManager? _libraryManager;
    public ObservableCollection<RunSummary> Reports { get; } = [];
    private bool InLibraryMode => _libraryManager != null && _imageId != null;

    private readonly string _hdrPath;
    private bool _hasCalibration;

    private readonly InferenceService _inferenceService;
    private readonly PdfReportService? _pdfReportService;
    private ModelPackage? _lastUsedPackage;
    private string _lastSummaryText = "";
    public OverlayViewModel Overlay { get; } = new();

    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _loadTcs = new();

    // -- States -- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLoading))]
    [NotifyPropertyChangedFor(nameof(IsError))]
    [NotifyPropertyChangedFor(nameof(IsReady))]
    private LoadingState _loadingState = LoadingState.Idle;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(MaxBandIndex))]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private HsiCube? _cube;

    [ObservableProperty] private DisplayOption _selectedDisplayMode = DisplayOption.Default;
    public static IReadOnlyList<DisplayOption> AvailableDisplayModes => DisplayOption.Presets;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private int _selectedBand;

    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private Bitmap? _currentBitmap;
    [ObservableProperty] private string _inferenceOutput = "";
    [ObservableProperty] private bool _isPixelSpectrumMode;
    [ObservableProperty] private int _selectedPixelX = -1;
    [ObservableProperty] private int _selectedPixelY = -1;
    [ObservableProperty] private bool _hasSelectedSpectrum;
    [ObservableProperty] private IList<Point> _pixelSpectrumPoints = new List<Point>();
    [ObservableProperty] private string _pixelSpectrumInfo = "Klikk på et piksel for spektral signatur";
    [ObservableProperty] private string _spectrumAxisMinLabel = "";
    [ObservableProperty] private string _spectrumAxisMaxLabel = "";
    [ObservableProperty] private string _spectrumValueMinLabel = "";
    [ObservableProperty] private string _spectrumValueMaxLabel = "";

    // -- Computed properties -- //
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "??";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;
    public bool IsPanEnabled => !IsPixelSpectrumMode;
    public double MinZoomLevel => 1.0;
    public double MaxZoomLevel => IsPixelSpectrumMode ? 1.0 : 50.0;
    public double SpectrumChartWidth => 320;
    public double SpectrumChartHeight => 120;

    // -- Property change handlers -- //
    public bool IsSpectralMode => SelectedDisplayMode.DisplayMode == DisplayMode.SpectralBand;
    partial void OnSelectedBandChanged(int value) => UpdateBitmap();

    partial void OnSelectedDisplayModeChanged(DisplayOption value)
    {
        OnPropertyChanged(nameof(IsSpectralMode));
        UpdateBitmap();
    }

    partial void OnIsPixelSpectrumModeChanged(bool value)
    {
        OnPropertyChanged(nameof(IsPanEnabled));
        OnPropertyChanged(nameof(MaxZoomLevel));
    }

    // -- Image loading on Initialization (delegates to ImageLoadingService) -- //
    private async Task LoadAsync()
    {
        try
        {
            LoadingState = LoadingState.Loading;
            var progress = new Progress<(string Status, double Progress)>(p =>
            {
                StatusMessage = p.Status;
                Progress = p.Progress;
            });

            var result = await ImageLoadingService.LoadAsync(_hdrPath, progress, _cts.Token);
            Cube = result.Cube;
            _hasCalibration = result.HasCalibration;


            LoadingState = LoadingState.Ready;
            StatusMessage = "Loading Complete";
            UpdateBitmap();
            TrySaveThumbnail(_cachedSyntheticRgb!);
        }
        catch (OperationCanceledException)
        {
            LoadingState = LoadingState.Idle;
            StatusMessage = "Operation Cancelled";
        }
        catch (Exception ex)
        {
            LoadingState = LoadingState.Error;
            StatusMessage = $"Failed to load: {ex.Message}";
        }
        finally
        {
            _loadTcs.TrySetResult();
        }
    }

    [ObservableProperty] private bool _hasPreprocessedCube;
    private PreprocessingResult? _cachedPreprocessing;
    private ModelPackage? _lastPackage;

    /// <summary>
    /// Runs inference using the set model package optional stride override.
    /// </summary>
    [RelayCommand(IncludeCancelCommand = true)]
    private async Task RunInference(CancellationToken ct)
    {
        if (Cube == null || string.IsNullOrEmpty(_hdrPath))
        {
            InferenceOutput = "No image loaded";
            return;
        }

        if (!_hasCalibration)
        {
            InferenceOutput =
                "Missing calibration: place darkReference.hdr and whiteReference.hdr in the scene folder and reopen.";
            return;
        }

        var package = _inferenceService.GetActivePackage();
        if (package == null)
        {
            InferenceOutput = "No model available. Import one via the Models page.";
            return;
        }

        try
        {
            var progress = new Progress<string>(s => { InferenceOutput = s; });

            // Perform preprocessing if fresh session or different modelPackage
            if (_cachedPreprocessing == null || _lastPackage != package)
            {
                InferenceOutput = "Performing preprocessing...";
                _cachedPreprocessing = await Task.Run(
                    () => PreprocessingService.RunFromCalibrated(Cube!, package.Manifest.Pipeline.Preprocessing), ct);
                _lastPackage = package;
                HasPreprocessedCube = _cachedPreprocessing.HasValue;
            }

            // Perform inference on preprocessed cube
            var runResult = await _inferenceService.RunAsync(
                _cachedPreprocessing.Value, package, progress, ct);

            _lastUsedPackage = package;
            _lastSummaryText = ClassificationResultMetrics.BuildReportSummaryText(runResult, package.Manifest.DisplayName);
            InferenceOutput = _lastSummaryText;

            Overlay.ApplyResult(runResult, Cube!.Samples, Cube!.Lines);
            ExportPdfCommand.NotifyCanExecuteChanged();
            await TryAutoSaveRunAsync(runResult, ct);
        }
        catch (OperationCanceledException)
        {
            InferenceOutput = "Cancelled.";
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Error: {ex.Message}";
        }
    }

    private bool CanRunInference() => Cube != null && _hasCalibration;

    private bool CanExportPdf() =>
        _pdfReportService != null
        && IsReady
        && Overlay.ClassificationResult != null
        && Overlay.ClassificationResult.DatePerformed != default;

    [RelayCommand(CanExecute = nameof(CanExportPdf))]
    private async Task ExportPdfAsync()
    {
        if (_pdfReportService == null) return;

        var ownerWindow = GetTopLevelAsWindow();
        if (ownerWindow == null) return;

        var optionsDialog = new ExportOptionsDialog();
        await optionsDialog.ShowDialog(ownerWindow);
        var options = optionsDialog.Result;
        if (options == null) return;

        var suggested = $"SpectralAssist_{DateTime.Now:yyyyMMdd_HHmmss}.pdf";
        var file = await ownerWindow.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export PDF report",
            DefaultExtension = "pdf",
            SuggestedFileName = suggested,
            FileTypeChoices = [new FilePickerFileType("PDF") { Patterns = ["*.pdf"] }]
        });
        if (file == null) return;

        try
        {
            StatusMessage = "Genererer PDF…";
            var pdfService = _pdfReportService;
            var pdfBytes = await Task.Run(() =>
            {
                var doc = BuildPdfReportDocument(options);
                using var ms = new MemoryStream();
                pdfService.Write(ms, doc);
                return ms.ToArray();
            });

            await using var outStream = await file.OpenWriteAsync();
            await outStream.WriteAsync(pdfBytes);
            StatusMessage = "PDF-eksport fullført";
        }
        catch (Exception ex)
        {
            StatusMessage = $"PDF-eksport mislyktes: {ex.Message}";
        }
    }

    private static Window? GetTopLevelAsWindow()
    {
        if (Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime { MainWindow: { } window })
            return window;
        return null;
    }

    private PdfReportDocument BuildPdfReportDocument(ExportOptions options)
    {
        if (Cube == null || Overlay.ClassificationResult == null)
            throw new InvalidOperationException("Cannot build PDF: missing cube or result.");

        var result = Overlay.ClassificationResult;
        if (result.DatePerformed == default)
            throw new InvalidOperationException("Cannot build PDF: inference metadata is missing.");

        var w = Cube.Samples;
        var h = Cube.Lines;
        var heatmap = HeatmapRenderer.BuildHeatmap(result, w, h);
        var colorMap = ColorMaps.All.GetValueOrDefault(options.ColorMapName, ColorMaps.GreenRed);
        var manifestDisplayName = _lastUsedPackage?.Manifest.DisplayName ?? "";
        var accuracy = _lastUsedPackage?.Manifest.Training.Metrics.Accuracy;
        var accDisplay = accuracy is { } a ? $"{a:P1}" : "—";

        var rgb = CubeRenderer.SyntheticRgbToBitmap(Cube, SyntheticRgbParameters.HistologyBalanced);
        WriteableBitmap? c0 = null, c1 = null, c2 = null;
        try
        {
            using var ol0 = HeatmapRenderer.RenderHeatmap(heatmap, w, h, colorMap, 0f);
            using var ol1 = HeatmapRenderer.RenderHeatmap(heatmap, w, h, colorMap, options.Overlay1Threshold);
            using var ol2 = HeatmapRenderer.RenderHeatmap(heatmap, w, h, colorMap, options.Overlay2Threshold);

            c0 = RgbOverlayComposer.Compose(rgb, ol0, options.Opacity);
            c1 = RgbOverlayComposer.Compose(rgb, ol1, options.Opacity);
            c2 = RgbOverlayComposer.Compose(rgb, ol2, options.Opacity);

            return new PdfReportDocument
            {
                InferenceCompletedAt = new DateTimeOffset(result.DatePerformed, TimeSpan.Zero),
                ExportedAt = DateTimeOffset.Now,
                ManifestDisplayName = manifestDisplayName,
                ModelNameFromResult = result.ModelName,
                AccuracyDisplay = accDisplay,
                ReportSummaryText = _lastSummaryText,
                SyntheticRgbPng = EncodeForPdf(rgb),
                Overlay0Png = EncodeForPdf(c0),
                Overlay50Png = EncodeForPdf(c1),
                Overlay80Png = EncodeForPdf(c2),
                Overlay1Threshold = options.Overlay1Threshold,
                Overlay2Threshold = options.Overlay2Threshold,
                OverlayOpacity = options.Opacity,
            };
        }
        finally
        {
            rgb.Dispose();
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

    // -- Display -- //
    private void UpdateBitmap()
    {
        if (Cube == null) return;

        CurrentBitmap = SelectedDisplayMode.DisplayMode switch
        {
            DisplayMode.SpectralBand => CubeRenderer.BandToBitmap(Cube, SelectedBand),
            DisplayMode.SyntheticRgb => GetCachedSyntheticRgb(Cube),
            DisplayMode.NearestBandRgb => CubeRenderer.RgbToBitmap(Cube,
                Cube.Header.FindClosestBand(630f),
                Cube.Header.FindClosestBand(530f),
                Cube.Header.FindClosestBand(460f)),
            _ => throw new ArgumentOutOfRangeException(nameof(DisplayMode))
        };
    }

    private Bitmap? _cachedSyntheticRgb;

    /// <summary>
    /// Returns the cached synthetic RGB bitmap, recomputing only initially.
    /// </summary>
    private Bitmap GetCachedSyntheticRgb(HsiCube cube)
    {
        _cachedSyntheticRgb ??= CubeRenderer.SyntheticRgbToBitmap(cube, SyntheticRgbParameters.HistologyBalanced);
        return _cachedSyntheticRgb;
    }

    public void SelectPixelSpectrumAt(int x, int y)
    {
        if (Cube == null || !IsPixelSpectrumMode) return;

        SelectedPixelX = x;
        SelectedPixelY = y;

        var spectrum = Cube.GetSpectrumAt(x, y);
        BuildSpectrumPlot(spectrum);
        HasSelectedSpectrum = true;
        PixelSpectrumInfo = $"Piksel ({x}, {y})";
    }

    private void BuildSpectrumPlot(float[] spectrum)
    {
        if (spectrum.Length == 0)
        {
            PixelSpectrumPoints = new List<Point>();
            return;
        }

        var wavelengths = Cube?.Header.WavelengthValues;
        var hasWavelengths = wavelengths != null && wavelengths.Length == spectrum.Length;

        var minValue = spectrum.Min();
        var maxValue = spectrum.Max();
        var valueRange = Math.Max(maxValue - minValue, 1e-8f);
        var width = SpectrumChartWidth;
        var height = SpectrumChartHeight;

        var points = new List<Point>(spectrum.Length);
        for (var i = 0; i < spectrum.Length; i++)
        {
            var t = spectrum.Length == 1 ? 0.0 : (double)i / (spectrum.Length - 1);
            var px = Math.Clamp(t * (width - 1), 0, width - 1);
            var py = Math.Clamp(height - 1 - ((spectrum[i] - minValue) / valueRange * (height - 1)), 0, height - 1);
            points.Add(new Point(px, py));
        }

        PixelSpectrumPoints = points;

        if (hasWavelengths)
        {
            SpectrumAxisMinLabel = $"{wavelengths![0]:F1} {WavelengthUnit}";
            SpectrumAxisMaxLabel = $"{wavelengths[^1]:F1} {WavelengthUnit}";
        }
        else
        {
            SpectrumAxisMinLabel = "Band 0";
            SpectrumAxisMaxLabel = $"Band {Math.Max(0, spectrum.Length - 1)}";
        }

        SpectrumValueMinLabel = minValue.ToString("F4", CultureInfo.InvariantCulture);
        SpectrumValueMaxLabel = maxValue.ToString("F4", CultureInfo.InvariantCulture);
    }

    // Persistence Logic _______________________________

    [ObservableProperty] private string? _activeRunId;

    /// <summary>
    /// Silently tries to save a thumbnail of the given bitmap.
    /// Only works when loading images through the library (in library mode).
    /// </summary>
    /// <param name="bitmap">The bitmap to save as a thumbnail</param>
    private void TrySaveThumbnail(Bitmap bitmap)
    {
        if (_libraryManager == null || _imageNode == null) return;

        ThumbnailService.TrySaveFromBitmap(_libraryManager.Root!, _imageNode.ImageId, bitmap);
        _libraryManager.NotifyImageUpdated(_imageNode);
    }

    private async Task TryAutoSaveRunAsync(ClassificationReport report, CancellationToken ct = default)
    {
        if (_libraryManager == null || string.IsNullOrEmpty(_imageId)) return;
        try
        {
            var summary = await _libraryManager.SaveRunAsync(_imageId, report, ct);
            Reports.Insert(0, summary);
            ActiveRunId = summary.RunId;
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Inference succeeded but save failed: {ex.Message}";
        }
    }


    [RelayCommand]
    private async Task LoadRun(RunSummary? summary)
    {
        if (summary == null || _libraryManager == null || string.IsNullOrEmpty(_imageId)) return;
        if (Cube == null)
        {
            InferenceOutput = "Image still loading…";
            return;
        }

        var report = await _libraryManager.LoadRunAsync(_imageId, summary.RunId, _cts.Token);
        if (report == null)
        {
            InferenceOutput = "Run file missing or unreadable.";
            return;
        }

        Overlay.ApplyResult(report, Cube.Samples, Cube.Lines);
        ActiveRunId = summary.RunId;
        InferenceOutput = $"Loaded report from {summary.DatePerformed:yyyy-MM-dd} ({summary.ModelName})";
    }

    /// <summary>
    /// Deletes a saved run from disk and the library manifest, and removes it from the
    /// runs list. If the deleted run was currently displayed, clears the overlay.
    /// </summary>
    [RelayCommand]
    private async Task DeleteRun(RunSummary? summary)
    {
        if (summary == null || _libraryManager == null || string.IsNullOrEmpty(_imageId)) return;
        try
        {
            await _libraryManager.DeleteRunAsync(_imageId, summary.RunId, _cts.Token);
            Reports.Remove(summary);
            if (ActiveRunId == summary.RunId)
            {
                ActiveRunId = null;
                Overlay.Clear();
            }
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Delete failed: {ex.Message}";
        }
    }


    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();
        Overlay.Clear();
        _cachedPreprocessing = null;
        _lastPackage = null;
        CurrentBitmap = null;
        _cachedSyntheticRgb?.Dispose();
        _cachedSyntheticRgb = null;
        Cube = null;
        GC.SuppressFinalize(this);
    }


    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ImageViewModel()
    {
        _hdrPath = "design.hdr";
        _inferenceService = null!;
        _pdfReportService = null;

        var dummyHeader = new HsiHeader
        {
            Description = "Preview Sample",
            Samples = 320,
            Lines = 240,
            Bands = 3,
            WavelengthUnit = "nm",
            WavelengthValues = [460f, 530f, 630f],
        };
        Cube = new HsiCube(dummyHeader, new float[320 * 240 * 3]);

        // Placeholder Bitmap
        CurrentBitmap = new WriteableBitmap(
            new Avalonia.PixelSize(320, 240),
            new Avalonia.Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Opaque);

        LoadingState = LoadingState.Ready;
        StatusMessage = "Design preview";
    }
}
