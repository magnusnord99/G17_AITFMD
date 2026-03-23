using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

public enum LoadingState { Idle, Loading, Ready, Error }
public enum DisplayMode { Rgb, Grayscale }

public partial class ImageViewModel : ViewModelBase, IDisposable
{
    private readonly CancellationTokenSource _cts = new();
    private readonly string _hdrPath;
    private readonly ModelLoader _loader = new();
    private readonly OnnxClassifier _onnxClassifier = new();

    public ImageViewModel(string hdrPath = "")
    {
        _hdrPath = hdrPath;
        RunInferenceCommand = new AsyncRelayCommand(RunInferenceAsync);
        _ = LoadAsync();
    }

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

    [ObservableProperty] private DisplayMode _currentDisplayMode = DisplayMode.Rgb;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private int _selectedBand;

    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private WriteableBitmap? _currentBitmap;
    
    
    // -- Classification -- //
    [ObservableProperty] private string _inferenceOutput = "";
    [ObservableProperty] private WriteableBitmap? _overlayBitmap;
    [ObservableProperty] private double _overlayOpacity = 0.5;
    [ObservableProperty] private bool _showOverlay = true;
    [ObservableProperty] private ClassificationResult? _classificationResult;
    
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "Unit";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;
    partial void OnSelectedBandChanged(int value) => UpdateBitmap();


    public bool IsGrayscale
    {
        get => CurrentDisplayMode == DisplayMode.Grayscale;
        set
        {
            CurrentDisplayMode = value ? DisplayMode.Grayscale : DisplayMode.Rgb;
            OnPropertyChanged();
        }
    }

    partial void OnCurrentDisplayModeChanged(DisplayMode value)
    {
        OnPropertyChanged(nameof(IsGrayscale));
        UpdateBitmap();
    }


    private async Task LoadAsync()
    {
        try
        {
            LoadingState = LoadingState.Loading;
            StatusMessage = "Reading header...";
            Progress = 0;

            // Step 1: Parse header and show info (should be fast)
            var header = HsiHeaderParser.Parse(_hdrPath);
            StatusMessage = $"{header.Samples}x{header.Lines}x{header.Bands} bands, {header.Interleave.ToUpper()}";

            // Step 2: Load and convert binary data into memory (heavy load)
            var progressReporter = new Progress<(float percent, int band)>(p =>
            {
                Progress = p.percent;
                StatusMessage = $"Loading image data... {p.percent:P0}";
            });
            Cube = await HsiCubeLoader.LoadAsync(header, progressReporter, _cts.Token);

            // Step 3: Calibrate if references exist
            StatusMessage = "Checking for calibration references...";
            var calibrated = await HsiCalibration.TryCalibrateAsync(_hdrPath, Cube);

            if (calibrated != null)
            {
                Cube = calibrated;
                StatusMessage = "Calibration complete...";
            }
            else
            {
                StatusMessage = "Calibration skipped...";
            }


            // Step 4: Show image
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
    }

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


    public IAsyncRelayCommand RunInferenceCommand { get; }

    private async Task RunInferenceAsync()
    {
        if (Cube == null)
        {
            InferenceOutput = "No image loaded";
            return;
        }

        try
        {
            InferenceOutput = "Running classification...";

            // Pick classifier: Swap between these as needed
            const string packagePath =
                @"E:\Dev\Projects\Rider\G17_AITFMD\GUI_G17_AITFMD\spectral-assist\SpectralAssist\Assets\exported_model";
            const string dummyJsonPath =
                @"E:\Dev\Projects\Rider\G17_AITFMD\GUI_G17_AITFMD\spectral-assist\SpectralAssist\Assets\prediction.json";
            
            var classifier = DummyClassifier.Random(32, 32, 12);
            //var classifier = DummyClassifier.FromJson(dummyJsonPath);
            //var classifier = new PythonClassifier(_hdrPath);
            //var classifier = CreateOnnxClassifier(packagePath);

            ClassificationResult = await Task.Run(() => classifier.ClassifyImageAsync(Cube));

            InferenceOutput = FormatResultSummary(ClassificationResult);
            ShowOverlay = true;
            RebuildOverlay();
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Error: {ex.Message}";
        }
    }

    private OnnxClassifier CreateOnnxClassifier(string packagePath)
    {
        var package = _loader.LoadPackage(packagePath);
        _onnxClassifier.SetModel(package);
        return _onnxClassifier;
    }

    private void RebuildOverlay()
    {
        if (ClassificationResult == null || Cube == null) return;

        OverlayBitmap = BitmapRenderer.ClassificationOverlay(
            Cube,
            ClassificationResult,
            ColorMaps.GreenRed);
    }

    private static string FormatResultSummary(ClassificationResult result)
    {
        var text = new System.Text.StringBuilder();
        text.AppendLine($"Model: {result.ModelName}");
        text.AppendLine($"Evaluated: {result.Evaluated} patches ({result.Skipped} skipped)");
        text.AppendLine();

        foreach (var pred in result.Predictions)
        {
            var className = result.Classes[pred.PredictedClass];
            text.AppendLine($"  ({pred.X},{pred.Y}): {className} ({pred.Confidence:P1})");
        }

        return text.ToString();
    }

    public void Dispose()
    {
        _onnxClassifier.Dispose();
        _cts.Cancel();
        _cts.Dispose();
        CurrentBitmap = null;
        Cube = null;
        GC.SuppressFinalize(this);
    }
}
