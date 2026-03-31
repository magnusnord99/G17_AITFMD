using System;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

public enum LoadingState { Idle, Loading, Ready, Error }
public enum DisplayMode { Rgb, Grayscale }

/// <summary>
/// Coordinator ViewModel for the image analysis view.
/// Delegates loading to <see cref="ImageLoadingService"/>,
/// inference to <see cref="InferenceService"/>,
/// and overlay state to <see cref="OverlayManager"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase, IDisposable
{
    private readonly string _hdrPath;
    private bool _hasCalibration;
    private HsiCube? _calibratedCube;

    private readonly ImageLoadingService _loadingService;
    private readonly InferenceService _inference;
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
    
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsGrayscale))]
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

    public ImageViewModel(string hdrPath, ImageLoadingService loadingService, InferenceService inference)
    {
        _hdrPath = hdrPath;
        _loadingService = loadingService;
        _inference = inference;
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
            _calibratedCube = result.CalibratedCube;
            _hasCalibration = result.HasCalibration;
            Cube = result.Cube;

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

    // -- Inference on Click (delegates to InferenceManager) -- //
    // ToDO: Add loadTCs into delegation to prevent early start before/during image loading
    [RelayCommand]
    private async Task RunInference()
    {
        if (Cube == null || string.IsNullOrEmpty(_hdrPath))
        {
            InferenceOutput = "No image loaded";
            return;
        }

        if (_calibratedCube == null || !_hasCalibration)
        {
            InferenceOutput =
                "Inference requires calibrated data (dark + white reference). " +
                "Place dark/white .hdr files in the same folder and reopen the scene.";
            return;
        }

        try
        {
            InferenceOutput = "Running preprocessing pipeline...";
            var running = true;
            var progress = new Progress<string>(s =>
            {
                if (running) InferenceOutput = s;
            });
            var (result, summary) = await _inference.RunAsync(_calibratedCube, progress, _cts.Token);
            running = false;

            InferenceOutput = summary;
            Overlay.ApplyResult(result, Cube!);
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

        CurrentBitmap = CurrentDisplayMode switch
        {
            DisplayMode.Grayscale => BitmapRenderer.BandToBitmap(Cube, SelectedBand),

            DisplayMode.Rgb => BitmapRenderer.RgbToBitmap(
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
        CurrentBitmap = null;
        Cube = null;
        _calibratedCube = null;
        GC.SuppressFinalize(this);
    }
}