using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Library;
using SpectralAssist.Services.Rendering;
using SpectralAssist.ViewModels.Components;

namespace SpectralAssist.ViewModels;

public enum LoadingState
{
    Idle, Loading, Ready, Error
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
        string hdrPath, InferenceService inferenceService)
    {
        _hdrPath = hdrPath;
        _inferenceService = inferenceService;
        _ = LoadAsync();
    }

    // Library mode constructor: library context aware with persistence
    public ImageViewModel(string hdrPath, InferenceService inferenceService, 
        string imageId, LibraryManager libraryManager) : this(hdrPath, inferenceService)
    {
        _imageId = imageId;
        _libraryManager = libraryManager;

        foreach (var r in libraryManager.FindImage(imageId)?.Runs ?? Enumerable.Empty<RunSummary>())
            Reports.Add(r);
    }
    
    private readonly string? _imageId;
    private readonly LibraryManager? _libraryManager;
    public ObservableCollection<RunSummary> Reports { get; } = [];
    public bool IsInLibrary => _libraryManager != null && _imageId != null;
    
    private readonly string _hdrPath;
    private bool _hasCalibration;
    
    private readonly InferenceService _inferenceService;
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

    // -- Computed properties -- //
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "??";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;

    // -- Property change handlers -- //
    public bool IsSpectralMode => SelectedDisplayMode.DisplayMode == DisplayMode.SpectralBand;
    partial void OnSelectedBandChanged(int value) => UpdateBitmap();

    partial void OnSelectedDisplayModeChanged(DisplayOption value)
    {
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

            var result = await ImageLoadingService.LoadAsync(_hdrPath, progress, _cts.Token);
            Cube = result.Cube;
            _hasCalibration = result.HasCalibration;


            LoadingState = LoadingState.Ready;
            StatusMessage = "Loading Complete";
            UpdateBitmap();
            TrySaveThumbnail(_cachedSyntheticRgb);
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
                "Missing calibration: place darkReference.hdr and whiteReference.hdr in the scene folder and reopen.";
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
            var classificationResult = await _inferenceService.RunAsync(
                _cachedPreprocessing.Value, modelPackage, stride, progress, _cts.Token);
            running = false;
            
            Overlay.ApplyResult(classificationResult, Cube!.Samples, Cube!.Lines);
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
    

    
    
    
    // --- Persistence: auto-save after inference, load on click, delete --- //
    
    /// <summary>
    /// Silently tries to save a thumbnail of the given bitmap.
    /// Only works when loading images through the library (in library mode).
    /// </summary>
    /// <param name="bitmap">The bitmap to save as a thumbnail</param>
    private void TrySaveThumbnail(Bitmap bitmap)
    {
        if (_libraryManager?.Root != null && !string.IsNullOrEmpty(_imageId))
            ThumbnailService.TrySaveFromBitmap(_libraryManager.Root, _imageId, bitmap);
    }
    
    [ObservableProperty] private string? _activeRunId;
    
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
        Cube = null;
        GC.SuppressFinalize(this);
    }

    
    

    // ToDo: Split view?
    // Export:
    // 4 ulike bilder: 
    // 1: RGB standard
    // 2: RGB med overlay?
    // 3: RGB med overlay med en viss threshold 80%?
    // 4: RGB med overlay med en viss threshold 50%?


    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ImageViewModel()
    {
        _hdrPath = "design.hdr";
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