using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Extensions;
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
        LibraryManager libraryManager)
    {
        ImageNode = imageNode;
        _inferenceService = inferenceService;
        _libraryManager = libraryManager;

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
    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _loadTcs = new();

    private PropertyChangedEventHandler? _overlayHandler;
    private bool _isCalibrated;
    private bool InLibraryMode => ImageNode.IsInLibrary;
    private bool HasCalibration => ImageNode.HasCalibration;
    private bool CanExportPdf() => Cube != null && Overlay.ClassificationResult != null;


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

            // Preprocess (cache invalidation by package change)
            if (_cachedPreprocessing == null || _lastPackage != package)
            {
                InferenceOutput = "Performing preprocessing...";
                _cachedPreprocessing = await Task.Run(
                    () => PreprocessingService.RunFromCalibrated(Cube!, package.Manifest.Pipeline.Preprocessing), ct);
                _lastPackage = package;
                HasPreprocessedCube = _cachedPreprocessing.HasValue;
            }

            // Perform Inference
            var runResult = await _inferenceService.RunAsync(
                _cachedPreprocessing.Value, package, progress, ct);
            
            Overlay.ApplyResult(runResult, Cube!.Samples, Cube!.Lines);
            var summary = await TryAutoSaveRunAsync(runResult, ct);
            if (summary != null)
                ActiveRun = summary;
            
            InferenceOutput = "";
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
            return await _libraryManager.SaveRunAsync(ImageNode.ImageId, report, ct);
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
        InferenceOutput = $"Loaded report from {summary.CompletedAt:yyyy-MM-dd HH:mm} ({summary.ModelDisplayName})";
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
        _cachedSyntheticRgb?.Dispose();

        if (_overlayHandler != null)
            Overlay.PropertyChanged -= _overlayHandler;
        Overlay.Clear();

        ActiveRun = null;
        Cube = null;
        CurrentBitmap = null;
        _cachedSyntheticRgb = null;
        _cachedPreprocessing = null;
        _lastPackage = null;

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
        if (Cube == null || Overlay.ClassificationResult == null || Overlay.CachedHeatmap == null) return;
        
        var ownerWindow = Application.Current?.MainWindow();
        if (ownerWindow == null) return;

        var defaults = ExportOptions.FromOverlay(
            Overlay.SelectedColorMapName, (float)Overlay.OverlayOpacity, (float)Overlay.OverlayThreshold);

        var dialog = new ExportDialog();
        dialog.DataContext = new ExportDialogViewModel(dialog, defaults);
        var options = await dialog.ShowDialog<ExportOptions?>(ownerWindow);
        if (options == null) return;

        var file = await ownerWindow.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export PDF Report",
            DefaultExtension = "pdf",
            SuggestedFileName = $"SpectralAssist_{DateTime.Now:yyyyMMdd_HHmmss}.pdf",
            FileTypeChoices = [new FilePickerFileType("PDF") { Patterns = ["*.pdf"] }]
        });
        if (file == null) return;

        try
        {
            var report = Overlay.ClassificationResult;
            var heatmap = Overlay.CachedHeatmap;
            var rgb = GetCachedSyntheticRgb(Cube, SyntheticRgbParameters.HistologyBalanced);

            StatusMessage = "Generating PDF...";
            var pdfBytes = await Task.Run(() =>
            {
                using var ms = new MemoryStream();
                PdfReportExporter.Export(report, rgb, heatmap, options, ms);
                return ms.ToArray();
            });

            await using var outStream = await file.OpenWriteAsync();
            await outStream.WriteAsync(pdfBytes);
            StatusMessage = "PDF export completed.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"PDF export failed: {ex.Message}";
        }
    }

    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ImageViewModel()
    {
        ImageNode = ImageNode.CreateTransient("design.hdr");
        _libraryManager = null!;
        _inferenceService = null!;

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