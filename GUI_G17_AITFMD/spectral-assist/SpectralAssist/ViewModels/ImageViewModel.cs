using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

public enum LoadingState
{
    Idle,
    Loading,
    Ready,
    Error
}

public enum DisplayMode
{
    Rgb,
    Grayscale
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

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(IsGrayscale))]
    private DisplayMode _currentDisplayMode = DisplayMode.Rgb;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private int _selectedBand;

    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private WriteableBitmap? _currentBitmap;
    [ObservableProperty] private string _inferenceOutput = "";

    // -- Computed properties -- //
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "??";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;

    public bool IsGrayscale
    {
        get => CurrentDisplayMode == DisplayMode.Grayscale;
        set => CurrentDisplayMode = value ? DisplayMode.Grayscale : DisplayMode.Rgb;
    }

    // -- Property change handlers -- //
    partial void OnSelectedBandChanged(int value) => UpdateBitmap();
    partial void OnCurrentDisplayModeChanged(DisplayMode value) => UpdateBitmap();

    public ImageViewModel(
        string hdrPath, 
        ImageLoadingService loadingService,
        InferenceService inferenceService)
    {
        _hdrPath = hdrPath;
        _loadingService = loadingService;
        _inferenceService = inferenceService;
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

    // -- Inference on Click (delegates to InferenceManager) -- //
    // ToDO: Add loadTCs into delegation to prevent early start before/during image loading
    /// <summary>
    /// Runs inference using the model at the given package directory.
    /// Called from MainViewModel which resolves the selected model.
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
                InferenceOutput = "Preprocessing (manifest-driven)...";
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

            InferenceOutput = FormatResultSummary(classificationResult);
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
    
    // -- Debug Summary (remove for production) -- //
    private static string FormatResultSummary(ClassificationResult result)
    {
        var text = new StringBuilder();
        text.AppendLine($"Model: {result.ModelName}");
        text.AppendLine($"Evaluated: {result.Evaluated} patches ({result.Skipped} skipped as background)");
        text.AppendLine();
 
        foreach (var pred in result.Predictions)
        {
            var className = result.Classes[pred.PredictedClass];
            text.AppendLine($"  ({pred.X},{pred.Y}): {className} ({pred.Confidence:P1})");
        }
 
        return text.ToString();
    }


    // -- Display -- //
    private void UpdateBitmap()
    {
        if (Cube == null) return;

        CurrentBitmap = CurrentDisplayMode switch
        {
            DisplayMode.Grayscale => CubeRenderer.BandToBitmap(Cube, SelectedBand),

            DisplayMode.Rgb => CubeRenderer.RgbToBitmap(
                Cube,
                Cube.Header.FindClosestBand(630f),
                Cube.Header.FindClosestBand(530f),
                Cube.Header.FindClosestBand(460f)),
            _ => CurrentBitmap
        };
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
    
    
    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ImageViewModel()
    {
        _hdrPath = "design.hdr";
        _loadingService = null!;
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