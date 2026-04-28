using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Primitives;
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
/// <item><see cref="PreprocessingService"/> for manifest-driven preprocessing</item>
/// <item><see cref="InferenceService"/> for ONNX model inference</item>
/// </list>
/// Overlay state is managed by <see cref="OverlayViewModel"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase, IDisposable
{

    public ImageViewModel(
        ImageNode imageNode,
        InferenceService inferenceService,
        LibraryManager libraryManager,
        PdfReportService pdfReportService)
    {
        ImageNode = imageNode;
        _inferenceService = inferenceService;
        _libraryManager = libraryManager;
        _pdfReportService = pdfReportService;
        
        // Add listener for overlay changes
        _overlayHandler = (_, e) =>
        {
            if (e.PropertyName == nameof(OverlayViewModel.ClassificationResult))
                ExportPdfCommand.NotifyCanExecuteChanged();
        };
        Overlay.PropertyChanged += _overlayHandler;
        
        _ = LoadAsync();
    }
    
    
    //private readonly ImageNode _imageNode;
    public ImageNode ImageNode { get; }
    public OverlayViewModel Overlay { get; } = new();
    
    private readonly InferenceService _inferenceService;
    private readonly LibraryManager _libraryManager;
    private readonly PdfReportService _pdfReportService;
    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _loadTcs = new();

    private PropertyChangedEventHandler? _overlayHandler;
    private bool _isCalibrated;
    private bool InLibraryMode => ImageNode.IsInLibrary;
    private bool HasCalibration => ImageNode.HasCalibration;
    private bool CanExportPdf() => ActiveRun != null && Cube != null;


    [ObservableProperty] [NotifyCanExecuteChangedFor(nameof(ExportPdfCommand))]
    private RunSummary? _activeRun;



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
    [NotifyPropertyChangedFor(nameof(ImageWidth))]
    [NotifyPropertyChangedFor(nameof(ImageHeight))]
    [NotifyCanExecuteChangedFor(nameof(ExportPdfCommand))]
    private HsiCube? _cube;

    [ObservableProperty] private DisplayOption _selectedDisplayMode = DisplayOption.Default;
    public static IReadOnlyList<DisplayOption> AvailableDisplayModes => DisplayOption.Presets;


    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private Bitmap? _currentBitmap;
    [ObservableProperty] private string _inferenceOutput = "";

    // -- Computed properties -- //
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "??";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;


    // DisplayMode Changes ==================================================================

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private int _selectedBand;

    partial void OnSelectedBandChanged(int value)
    {
        if (IsSpectralMode) UpdateBitmap();
    }

    public bool IsRgbMode
    {
        get => SelectedDisplayMode.DisplayMode == DisplayMode.SyntheticRgb;
        set
        {
            if (value)
                SelectedDisplayMode = DisplayOption.Presets.First(p => p.DisplayMode == DisplayMode.SyntheticRgb);
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsSpectralMode));
        }
    }

    public bool IsSpectralMode
    {
        get => SelectedDisplayMode.DisplayMode == DisplayMode.SpectralBand;
        set
        {
            if (value)
                SelectedDisplayMode = DisplayOption.Presets.First(p => p.DisplayMode == DisplayMode.SpectralBand);
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsRgbMode));
        }
    }

    partial void OnSelectedDisplayModeChanged(DisplayOption value)
    {
        OnPropertyChanged(nameof(IsRgbMode));
        OnPropertyChanged(nameof(IsSpectralMode));
        UpdateBitmap();
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

            var result = await ImageLoadingService.LoadAsync(ImageNode.AbsolutePath, progress, _cts.Token);
            Cube = result.Cube;
            _isCalibrated = result.HasCalibration;

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
            Console.WriteLine("Error Loading Image: " + ex.Message);
            Console.WriteLine(ex);
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
        if (Cube == null)
        {
            InferenceOutput = "No image loaded";
            return;
        }

        if (!_isCalibrated)
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
            InferenceOutput = "";

            Overlay.ApplyResult(runResult, Cube!.Samples, Cube!.Lines);
            
            var summary = await TryAutoSaveRunAsync(runResult, ct);
            if (summary != null)
                ActiveRun = summary;
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

    // Display and Bitmaps ____________________________________________________
    private Bitmap? _cachedSyntheticRgb;

    private void UpdateBitmap()
    {
        if (Cube == null) return;
        var option = SelectedDisplayMode;

        CurrentBitmap = option.DisplayMode switch
        {
            DisplayMode.SpectralBand => CubeRenderer.BandToBitmap(Cube, SelectedBand),
            DisplayMode.SyntheticRgb => GetCachedSyntheticRgb(Cube, option.RgbParameters),
            _ => throw new ArgumentOutOfRangeException(nameof(option.DisplayMode))
        };
    }

    /// <summary>
    /// Returns the cached synthetic RGB bitmap, recomputing only initially.
    /// Returns a synthetic RGB bitmap for the given parameters, computing
    /// and caching it on first access. Subsequent calls with the same
    /// parameters return the cached bitmap without recomputation.
    /// </summary>
    private Bitmap GetCachedSyntheticRgb(HsiCube cube, SyntheticRgbParameters parameters)
    {
        if (_cachedSyntheticRgb is not null)
            return _cachedSyntheticRgb;

        var bitmap = CubeRenderer.SyntheticRgbToBitmap(cube, parameters);
        _cachedSyntheticRgb = bitmap;
        return bitmap;
    }


    // Persistence Logic _______________________________
    
    /// <summary>
    /// Silently tries to save a thumbnail of the given bitmap.
    /// Only works when loading images through the library (in library mode).
    /// </summary>
    /// <param name="bitmap">The bitmap to save as a thumbnail</param>
    private void TrySaveThumbnail(Bitmap bitmap)
    {
        if (!InLibraryMode) return;
        
        ThumbnailService.TrySaveFromBitmap(_libraryManager.Root!, ImageNode.ImageId, bitmap);
        _libraryManager.NotifyImageUpdated(ImageNode);
    }

    private async Task<RunSummary?> TryAutoSaveRunAsync(ClassificationReport report, CancellationToken ct = default)
    {
        if (!InLibraryMode) return null;
        try
        {
            var summary = await _libraryManager.SaveRunAsync(ImageNode.ImageId, report, ct);
            return summary;
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Inference succeeded but save failed: {ex.Message}";
            return null;
        }
    }


    [RelayCommand]
    private async Task LoadRun(RunSummary? summary)
    {
        if (summary == null || !InLibraryMode) return;
        
        if (Cube == null)
        {
            InferenceOutput = "Image still loading...";
            return;
        }

        var report = await _libraryManager.LoadRunAsync(ImageNode.ImageId, summary.RunId, _cts.Token);
        if (report == null)
        {
            InferenceOutput = "Run file missing or unreadable.";
            return;
        }

        Overlay.ApplyResult(report, Cube.Samples, Cube.Lines);
        ActiveRun = summary;
        InferenceOutput = $"Loaded report from {summary.DatePerformed:yyyy-MM-dd HH:mm} ({summary.ModelDisplayName})";
    }

    /// <summary>
    /// Deletes a saved run from disk and the library manifest, and removes it from the
    /// runs list. If the deleted run was currently displayed, clears the overlay.
    /// </summary>
    [RelayCommand]
    private async Task DeleteRun(RunSummary? summary)
    {
        if (summary == null || !InLibraryMode) return;
        try
        {
            await _libraryManager.DeleteRunAsync(ImageNode.ImageId, summary.RunId, _cts.Token);
            if (ActiveRun?.RunId == summary.RunId)
            {
                ActiveRun = null;
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
        
        if (_overlayHandler != null)
            Overlay.PropertyChanged -= _overlayHandler;
        Overlay.Clear();
        
        _cachedPreprocessing = null;
        _lastPackage = null;
        CurrentBitmap = null;
        _cachedSyntheticRgb?.Dispose();
        _cachedSyntheticRgb = null;
        Cube = null;
        GC.SuppressFinalize(this);
    }


    public int ImageWidth => Cube?.Samples ?? 0;
    public int ImageHeight => Cube?.Lines ?? 0;
    [ObservableProperty] private bool _isSplitViewEnabled;


    // Spectral Signature _______________________________________________

    [ObservableProperty] private int? _pixel1X;
    [ObservableProperty] private int? _pixel1Y;
    [ObservableProperty] private int? _pixel2X;
    [ObservableProperty] private int? _pixel2Y;
    public bool HasAnySelection => Pixel1X is not null || Pixel2X is not null;

    public void OnPixelClicked(int x, int y, bool isPrimary)
    {
        if (Cube == null || x < 0 || y < 0 || x >= Cube.Samples || y >= Cube.Lines)
            return;

        if (isPrimary)
        {
            Pixel1X = x;
            Pixel1Y = y;
        }
        else
        {
            Pixel2X = x;
            Pixel2Y = y;
        }

        OnPropertyChanged(nameof(HasAnySelection));
    }

    public void ClearPixelSelections()
    {
        Pixel1X = null;
        Pixel1Y = null;
        Pixel2X = null;
        Pixel2Y = null;
        OnPropertyChanged(nameof(HasAnySelection));
    }


    // PDF Export _________________________________________
    
    [RelayCommand(CanExecute = nameof(CanExportPdf))]
    private async Task ExportPdfAsync()
    {
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
        if (Avalonia.Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime
            {
                MainWindow: { } window
            })
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
        //var manifestDisplayName = Pack._lastUsedPackage?.Manifest.DisplayName ?? "";
        //var accuracy = _lastUsedPackage?.Manifest.Training.Metrics.Accuracy;
        //var accDisplay = accuracy is { } a ? $"{a:P1}" : "—";

        var rgb = CubeRenderer.SyntheticRgbToBitmap(Cube, SyntheticRgbParameters.HistologyBalanced);
        Bitmap? c0 = null, c1 = null, c2 = null;
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
                ManifestDisplayName = "dummy name",
                ModelNameFromResult = result.ModelDisplayName,
                AccuracyDisplay = "very nice 9999%",
                ReportSummaryText = "THIS MUST BE FIXED PLS",
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


    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ImageViewModel()
    {
        ImageNode = ImageNode.CreateTransient("design.hdr");
        _libraryManager = null!;
        _inferenceService = null!;
        _pdfReportService = null!;

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