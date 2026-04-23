using System;
using System.Collections.Generic;
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
using SpectralAssist.Services.Rendering;
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
/// Overlay state is managed by <see cref="OverlayManager"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase, IDisposable
{
    private readonly string _hdrPath;
    private bool _hasCalibration;
    
    private readonly ImageLoadingService _loadingService;
    private readonly InferenceService _inferenceService;
    private readonly PdfReportService? _pdfReportService;
    public OverlayManager Overlay { get; } = new();

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
    [ObservableProperty] private WriteableBitmap? _currentBitmap;
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
    
    public ImageViewModel(
        string hdrPath,
        ImageLoadingService loadingService,
        InferenceService inferenceService,
        PdfReportService? pdfReportService = null)
    {
        _hdrPath = hdrPath;
        _loadingService = loadingService;
        _inferenceService = inferenceService;
        _pdfReportService = pdfReportService;
        _ = LoadAsync();
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
    private string _lastSummaryText = "";
    
    /// <summary>
    /// Runs inference using the given model package and the chosen stride.
    /// Invoked by the MainViewModel when inference button is clicked.
    /// </summary>
    public async Task RunInference(ModelPackage modelPackage, int stride)
    {
        if (Cube == null || string.IsNullOrEmpty(_hdrPath))
        {
            InferenceOutput = "No image loaded";
            return;
        }

        if (!_hasCalibration)
        {
            InferenceOutput =
                "Inference requires calibrated data (dark + white reference). " +
                "Place dark/white .hdr files in the same folder and reopen the scene.";
            return;
        }

        try
        {
            var running = true;
            var progress = new Progress<string>(s =>
            {
                if (running) InferenceOutput = s;
            });
            
            // Perform preprocessing if fresh session or different modelPackage
            if (_cachedPreprocessing == null || _lastPackage != modelPackage)
            {
                InferenceOutput = "Performing preprocessing...";
                var preprocessing = modelPackage.Manifest.Pipeline.Preprocessing;
                _cachedPreprocessing = await Task.Run(
                    () => PreprocessingService.RunFromCalibrated(Cube!, preprocessing), _cts.Token);
                _lastPackage = modelPackage;
                HasPreprocessedCube = _cachedPreprocessing.HasValue;
            }
            else
            {
                InferenceOutput = "Using cached preprocessing...";
            }
            
            // Perform inference on preprocessed cube
            var rawResult = await _inferenceService.RunAsync(
                _cachedPreprocessing.Value, modelPackage, stride, progress, _cts.Token);
            running = false;

            var classificationResult = WithInferenceReportMetadata(rawResult, modelPackage);
            _lastSummaryText = ClassificationResultMetrics.BuildReportSummaryText(classificationResult);
            InferenceOutput = _lastSummaryText;
            Overlay.ApplyResult(classificationResult, Cube!.Samples, Cube!.Lines);
            ExportPdfCommand.NotifyCanExecuteChanged();
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
    
    private static ClassificationResult WithInferenceReportMetadata(
        ClassificationResult raw,
        ModelPackage package)
    {
        var m = package.Manifest;
        return new ClassificationResult
        {
            Predictions = raw.Predictions,
            ImageWidth = raw.ImageWidth,
            ImageHeight = raw.ImageHeight,
            PatchH = raw.PatchH,
            PatchW = raw.PatchW,
            StrideH = raw.StrideH,
            StrideW = raw.StrideW,
            Classes = raw.Classes,
            ModelName = raw.ModelName,
            TotalPossible = raw.TotalPossible,
            Evaluated = raw.Evaluated,
            Skipped = raw.Skipped,
            ExecutionProvider = raw.ExecutionProvider,
            InferenceCompletedAt = DateTimeOffset.Now,
            ManifestDisplayName = m.DisplayName,
            TrainingValidationAccuracy = m.Training.Metrics.Accuracy,
        };
    }

    private bool CanExportPdf() =>
        _pdfReportService != null
        && IsReady
        && Overlay.ClassificationResult != null
        && Overlay.ClassificationResult.InferenceCompletedAt != default;

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
        if (result.InferenceCompletedAt == default)
            throw new InvalidOperationException("Cannot build PDF: inference metadata is missing.");
        var w = Cube.Samples;
        var h = Cube.Lines;
        var heatmap = HeatmapRenderer.BuildHeatmap(result, w, h);
        var colorMap = ColorMaps.All.GetValueOrDefault(options.ColorMapName, ColorMaps.GreenRed);

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

            var accDisplay = result.TrainingValidationAccuracy is { } a ? $"{a:P1}" : "—";

            return new PdfReportDocument
            {
                InferenceCompletedAt = result.InferenceCompletedAt,
                ExportedAt = DateTimeOffset.Now,
                ManifestDisplayName = result.ManifestDisplayName,
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
    
    private WriteableBitmap? _cachedSyntheticRgb;
    
    /// <summary>
    /// Returns the cached synthetic RGB bitmap, recomputing only initially.
    /// </summary>
    private WriteableBitmap GetCachedSyntheticRgb(HsiCube cube)
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
        _loadingService = null!;
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