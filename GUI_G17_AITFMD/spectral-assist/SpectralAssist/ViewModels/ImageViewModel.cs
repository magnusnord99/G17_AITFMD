using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.ViewModels;

public enum LoadingState { Idle, Loading, Ready, Error }
public enum DisplayMode { Rgb, Grayscale }

public partial class ImageViewModel : ViewModelBase, IDisposable
{
    private readonly CancellationTokenSource _cts = new();
    private readonly string _hdrPath;
    private readonly ModelLoader _loader = new();
    private readonly Onnx3DCnnClassifier _onnx3DCnn = new();
    private string? _loadedModelPackageDir;

    /// <summary>Rå scene + dark/white fra forrige Open — gjenbrukes ved inferens (ingen ny lasting fra disk).</summary>
    private HsiCube? _pipelineRaw;
    private HsiCube? _pipelineDark;
    private HsiCube? _pipelineWhite;

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
            var scene = await HsiCubeLoader.LoadAsync(header, progressReporter, _cts.Token);
            _pipelineRaw = null;
            _pipelineDark = null;
            _pipelineWhite = null;

            if (HsiCalibration.TryFindReferenceHdrPaths(_hdrPath, out var darkHdr, out var whiteHdr))
            {
                StatusMessage = "Loading dark/white references...";
                var darkTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(darkHdr!), ct: _cts.Token);
                var whiteTask = HsiCubeLoader.LoadAsync(HsiHeaderParser.Parse(whiteHdr!), ct: _cts.Token);
                await Task.WhenAll(darkTask, whiteTask);
                var dark = darkTask.Result;
                var white = whiteTask.Result;

                if (scene.Bands != dark.Bands || scene.Bands != white.Bands)
                {
                    LoadingState = LoadingState.Error;
                    StatusMessage =
                        $"Band mismatch: scene {scene.Bands}, dark {dark.Bands}, white {white.Bands}";
                    Cube = scene;
                    return;
                }

                _pipelineRaw = scene;
                _pipelineDark = dark;
                _pipelineWhite = white;
                Cube = HsiCalibration.ApplyReflectance(scene, dark, white);
                StatusMessage = "Calibration complete...";
            }
            else
            {
                Cube = scene;
                StatusMessage = "Calibration skipped (no dark/white in folder)...";
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
        if (Cube == null || string.IsNullOrEmpty(_hdrPath))
        {
            InferenceOutput = "No image loaded";
            return;
        }

        try
        {
            InferenceOutput = "Forbereder pipeline (rå → kalibrering → avg3 → maske → 16 bånd)...";

            if (_pipelineRaw == null || _pipelineDark == null || _pipelineWhite == null)
            {
                InferenceOutput =
                    "Pipeline trenger rå scene pluss dark og white lastet ved Open. " +
                    "Legg dark/white .hdr i samme mappe og åpne scenen på nytt.";
                return;
            }

            var packageDir = ResolveModelPackageDirectory();
            if (!Directory.Exists(packageDir))
            {
                InferenceOutput =
                    $"Model mappe ikke funnet:\n{packageDir}\n\nLegg manifest.json + model.onnx under Assets/models/... og bygg på nytt.";
                return;
            }

            StatusMessage = "Spektral pipeline (kalibrering → clip → avg3 → vevsmaske → 16 bånd)...";
            var pipelineResult = await Task.Run(() =>
            {
                var rf = HsiCubeToFloatCubeHWB.FromHsiCube(_pipelineRaw);
                var df = HsiCubeToFloatCubeHWB.FromHsiCube(_pipelineDark);
                var wf = HsiCubeToFloatCubeHWB.FromHsiCube(_pipelineWhite);
                var spectral = new BaselinePreprocessingOptions(
                    calibrationEpsilon: 1e-8f,
                    clipMin: 0f,
                    clipMax: 1f,
                    neighborAverageWindow: 3,
                    bandReduceOutBands: 16,
                    bandReduceStrategy: "crop");
                var tissue = new TissueMaskMeanStdOptions(
                    qMean: 0.5f,
                    qStd: 0.4f,
                    minObjectSize: 500,
                    minHoleSize: 1000);
                return BaselineSpectralPipeline.RunThroughAvg16WithTissueMask(rf, df, wf, spectral, tissue);
            }, _cts.Token);

            var hsi16 = FloatCubeToHsiCube.ToHsiCube(pipelineResult.Cube16Bands);

            StatusMessage = "ONNX 3D-CNN...";
            var classifier = EnsureOnnx3DClassifier(packageDir);
            ClassificationResult = await classifier.ClassifyImageAsync(hsi16);

            InferenceOutput = FormatResultSummary(ClassificationResult, pcaTrainingMismatchNote: true);
            ShowOverlay = true;
            RebuildOverlay();
            StatusMessage = "Inferens ferdig";
        }
        catch (OperationCanceledException)
        {
            InferenceOutput = "Avbrutt.";
        }
        catch (Exception ex)
        {
            InferenceOutput = $"Error: {ex.Message}";
        }
    }

    /// <summary>ONNX-pakke under output-katalog (Assets kopieres hit ved build).</summary>
    private static string ResolveModelPackageDirectory()
    {
        return Path.Combine(
            AppContext.BaseDirectory,
            "Assets",
            "models",
            "baseline_3dcnn_20260324_083658_last");
    }

    private Onnx3DCnnClassifier EnsureOnnx3DClassifier(string packageDir)
    {
        var fullPath = Path.GetFullPath(packageDir);
        if (_loadedModelPackageDir == fullPath && _onnx3DCnn.Manifest != null)
            return _onnx3DCnn;

        var package = _loader.LoadPackage(fullPath);
        _onnx3DCnn.SetModel(package);
        _loadedModelPackageDir = fullPath;
        return _onnx3DCnn;
    }

    private void RebuildOverlay()
    {
        if (ClassificationResult == null || Cube == null) return;

        OverlayBitmap = BitmapRenderer.ClassificationOverlay(
            Cube,
            ClassificationResult,
            ColorMaps.GreenRed);
    }

    private static string FormatResultSummary(ClassificationResult result, bool pcaTrainingMismatchNote = false)
    {
        var text = new System.Text.StringBuilder();
        if (pcaTrainingMismatchNote)
        {
            text.AppendLine(
                "Merk: Modellen er trent på PCA-16 (Python); GUI bruker band-gjennomsnitt til 16 — " +
                "fordelingen kan avvike. For best match, tren på samme reduksjon som i app.");
            text.AppendLine();
        }

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
        _onnx3DCnn.Dispose();
        _loader.Dispose();
        _cts.Cancel();
        _cts.Dispose();
        CurrentBitmap = null;
        Cube = null;
        _pipelineRaw = null;
        _pipelineDark = null;
        _pipelineWhite = null;
        GC.SuppressFinalize(this);
    }
}
