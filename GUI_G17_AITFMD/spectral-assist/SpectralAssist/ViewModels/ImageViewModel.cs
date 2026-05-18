using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
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
using SpectralAssist.Services.Hsi;
using SpectralAssist.Services.Inference;
using SpectralAssist.Services.Library;
using SpectralAssist.Services.Preprocessing;
using SpectralAssist.Services.Rendering;
using SpectralAssist.ViewModels.Components;

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
/// <item><see cref="ImageLoader"/> loads and calibrates the HSI cube</item>
/// <item><see cref="PreprocessingPipeline"/> for manifest-driven preprocessing</item>
/// <item><see cref="InferenceService"/> for ONNX model inference</item>
/// </list>
/// Overlay state is managed by <see cref="OverlayViewModel"/>.
/// </summary>
public partial class ImageViewModel : ViewModelBase, IDisposable
{
    public ImageNode ImageNode { get; }
    public OverlayViewModel Overlay { get; } = new();

    private readonly InferenceService _inferenceService;
    private readonly LibraryManager _libraryManager;
    private readonly SessionService _session;
    private readonly IDialogService _dialogService;
    private readonly CancellationTokenSource _cts = new();
    private readonly TaskCompletionSource _loadTcs = new();
    private readonly PropertyChangedEventHandler? _overlayHandler;
    private readonly PropertyChangedEventHandler? _sessionHandler;

    private bool InLibraryMode => ImageNode.IsInLibrary;

    public ImageViewModel(
        ImageNode imageNode,
        InferenceService inferenceService,
        LibraryManager libraryManager,
        SessionService session,
        IDialogService dialogService)
    {
        ImageNode = imageNode;
        _inferenceService = inferenceService;
        _libraryManager = libraryManager;
        _session = session;
        _dialogService = dialogService;
        _imageNotes = imageNode.Notes;

        // Add listener for overlay changes
        _overlayHandler = (_, e) =>
        {
            if (e.PropertyName == nameof(OverlayViewModel.ClassificationResult))
                ExportPdfCommand.NotifyCanExecuteChanged();
        };
        Overlay.PropertyChanged += _overlayHandler;

        // Add listener for sessionHandler changes (active model)
        _sessionHandler = (_, e) =>
        {
            if (e.PropertyName == nameof(SessionService.ActiveModel))
            {
                OnPropertyChanged(nameof(RunBlockedReason));
                OnPropertyChanged(nameof(CanRunInference));
                RunInferenceCommand.NotifyCanExecuteChanged();
            }
        };
        _session.PropertyChanged += _sessionHandler;

        RebuildRunCards();
        ImageNode.Runs.CollectionChanged += OnRunsCollectionChanged;
        _ = LoadAsync();
    }

    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();
        _cachedSyntheticRgb?.Dispose();

        if (_overlayHandler != null)
            Overlay.PropertyChanged -= _overlayHandler;
        if (_sessionHandler != null)
            _session.PropertyChanged -= _sessionHandler;
        ImageNode.Runs.CollectionChanged -= OnRunsCollectionChanged;
        Overlay.Clear();

        ActiveRun = null;
        Cube = null;
        CurrentBitmap = null;
        _cachedSyntheticRgb = null;
        _cachedPreprocessing = null;
        _lastPackage = null;

        GC.SuppressFinalize(this);
    }


    public event Action? CloseRequested;

    /// <summary>
    /// Cancels the in-progress load and asks the host to close the view.
    /// </summary>
    [RelayCommand]
    private void CancelLoad()
    {
        _cts.Cancel();
        CloseRequested?.Invoke();
    }


    // -- Loading (delegates to ImageLoader) -------------------------------------- //

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
    [NotifyPropertyChangedFor(nameof(RunBlockedReason))]
    [NotifyPropertyChangedFor(nameof(CanRunInference))]
    [NotifyCanExecuteChangedFor(nameof(ExportPdfCommand))]
    [NotifyCanExecuteChangedFor(nameof(RunInferenceCommand))]
    private HsiCube? _cube;

    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private string _loadingMetadata = "";

    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;

    public int ImageWidth => Cube?.Samples ?? 0;
    public int ImageHeight => Cube?.Lines ?? 0;


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

            var result = await ImageLoader.LoadAsync(ImageNode.AbsolutePath, progress,
                onHeaderParsed: h =>
                    LoadingMetadata =
                        $"{h.Samples} height  ·  {h.Lines} width  ·  {h.Bands} bands  ·  {h.Interleave.ToUpperInvariant()}  ·  {h.DataTypeName}",
                _cts.Token);
            Cube = result.Cube;
            IsCalibrated = result.HasCalibration;

            var rgbParams = SelectedDisplayMode.RgbParameters;
            var preview = await Task.Run(
                () => CubeRenderer.SyntheticRgbToBitmap(result.Cube, rgbParams), _cts.Token);
            _cachedSyntheticRgb = preview;

            LoadingState = LoadingState.Ready;
            UpdateBitmap();

            if (InLibraryMode)
            {
                var bmp = preview;
                _ = Task.Run(() => TrySaveThumbnail(bmp), _cts.Token);
            }
        }
        catch (OperationCanceledException)
        {
            LoadingState = LoadingState.Idle;
            StatusMessage = "Operation Cancelled";
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error Loading Image: " + ex.Message);
            LoadingState = LoadingState.Error;
            StatusMessage = $"Failed to load: {ex.Message}";
        }
        finally
        {
            _loadTcs.TrySetResult();
        }
    }


    // -- Display Mode & Bitmaps (RGB + Spectral Band) ----------------------- //

    [ObservableProperty] private DisplayOption _selectedDisplayMode = DisplayOption.Default;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WavelengthUnit))]
    [NotifyPropertyChangedFor(nameof(SelectedBandWaveLength))]
    private int _selectedBand;

    [ObservableProperty] private Bitmap? _currentBitmap;
    [ObservableProperty] private bool _isSplitViewEnabled;

    private Bitmap? _cachedSyntheticRgb;

    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string WavelengthUnit => Cube?.Header.WavelengthUnit ?? "??";
    public float SelectedBandWaveLength => Cube?.Header.WavelengthValues[SelectedBand] ?? -1f;

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

    partial void OnSelectedBandChanged(int value)
    {
        if (IsSpectralMode) UpdateBitmap();
    }

    partial void OnSelectedDisplayModeChanged(DisplayOption value)
    {
        OnPropertyChanged(nameof(IsRgbMode));
        OnPropertyChanged(nameof(IsSpectralMode));
        UpdateBitmap();
    }

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
    /// Returns the cached synthetic RGB bitmap, computing it on first access
    /// and reusing it on subsequent calls.
    /// </summary>
    private Bitmap GetCachedSyntheticRgb(HsiCube cube, SyntheticRgbParameters parameters)
    {
        if (_cachedSyntheticRgb is not null)
            return _cachedSyntheticRgb;

        var bitmap = CubeRenderer.SyntheticRgbToBitmap(cube, parameters);
        _cachedSyntheticRgb = bitmap;
        return bitmap;
    }


    // -- Inference (preprocess + ONNX run) -------------------------------- //

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(RunBlockedReason))]
    [NotifyPropertyChangedFor(nameof(CanRunInference))]
    [NotifyCanExecuteChangedFor(nameof(RunInferenceCommand))]
    private bool _isCalibrated;

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(RunStatusText))]
    private double _inferenceProgress;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPreprocessingPhase))]
    [NotifyPropertyChangedFor(nameof(RunStatusText))]
    private string? _inferencePhase;

    [ObservableProperty] private string? _lastRunError;
    [ObservableProperty] private bool _hasPreprocessedCube;

    private PreprocessingResult? _cachedPreprocessing;
    private ModelPackage? _lastPackage;

    /// <summary>True while the model package is being preprocessed (indeterminate progress).</summary>
    public bool IsPreprocessingPhase => InferencePhase == "Preprocessing";

    /// <summary>Human-readable status shown on the Run button while running.</summary>
    public string RunStatusText => InferencePhase switch
    {
        "Inferring" => $"Inferring  {InferenceProgress:P0}",
        "Preprocessing" => "Preprocessing…",
        _ => string.Empty,
    };

    /// <summary>
    /// Reason the user cannot run inference right now, or null if everything's ready.
    /// Drives the "Run Inference" button label + enabled state.
    /// </summary>
    public string? RunBlockedReason
    {
        get
        {
            if (Cube == null) return "Image still loading";
            if (!IsCalibrated) return "Missing calibration";
            if (_session.ActiveModel == null) return "No model selected";
            return null;
        }
    }

    public bool CanRunInference => RunBlockedReason == null;

    /// <summary>
    /// Runs inference using the set model package optional stride override.
    /// </summary>
    [RelayCommand(IncludeCancelCommand = true, CanExecute = nameof(CanRunInference))]
    private async Task RunInference(CancellationToken ct)
    {
        var package = _inferenceService.GetActivePackage();
        if (Cube == null || !IsCalibrated || package == null)
        {
            // Should not happen given CanExecute, but guard regardless.
            LastRunError = RunBlockedReason ?? "Cannot run inference";
            return;
        }

        LastRunError = null;
        InferenceProgress = 0;

        try
        {
            // Resource logging: setup up timers
            var wallTimer = Stopwatch.StartNew();
            var cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
            var preprocessingMs = 0.0;

            // Preform Preprocessing (cache invalidation by package change)
            if (_cachedPreprocessing == null || _lastPackage != package)
            {
                InferencePhase = "Preprocessing";
                var preprocessingTimer = Stopwatch.StartNew();
                _cachedPreprocessing = await Task.Run(
                    () => PreprocessingPipeline.RunFromCalibrated(Cube!, package.Manifest.Pipeline.Preprocessing, ct),
                    ct);

                preprocessingMs = preprocessingTimer.Elapsed.TotalMilliseconds;

                _lastPackage = package;
                HasPreprocessedCube = _cachedPreprocessing.HasValue;
            }

            // Perform Inference
            InferencePhase = "Inferring";
            var patchProgress = new Progress<(int Done, int Total)>(p =>
            {
                InferenceProgress = p.Total > 0 ? (double)p.Done / p.Total : 0;
            });

            var inferenceTimer = Stopwatch.StartNew();
            var runResult = await _inferenceService.RunAsync(
                _cachedPreprocessing.Value, package, patchProgress, ct);

            var inferenceMs = inferenceTimer.Elapsed.TotalMilliseconds;
            wallTimer.Stop();

            Overlay.ApplyResult(runResult, Cube!.Samples, Cube!.Lines);
            var summary = await TryAutoSaveRunAsync(runResult, ct);
            if (summary != null)
            {
                ActiveRun = summary;
            }

            // Resource logging: write one CSV row to console if flagged
            if (LogMetrics)
                LogMetricsCsv(
                    preprocessingMs,
                    inferenceMs,
                    wallTimer.Elapsed.TotalMilliseconds,
                    (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalMilliseconds,
                    Cube!,
                    _cachedPreprocessing.Value.Cube);
        }
        catch (OperationCanceledException)
        {
            LastRunError = "Cancelled";
        }
        catch (Exception ex)
        {
            LastRunError = $"Error: {ex.Message}";
        }
        finally
        {
            InferencePhase = null;
            InferenceProgress = 0;
        }
    }

    // -- Resource logging (CSV row per run; const flag toggles output) -- //

    private const bool LogMetrics = true;
    private static bool _headerPrinted;

    /// <summary>
    /// Writes one CSV row to the console per inference run, CPU timings and
    /// cube-size compression can be aggregated for the performance evaluation.
    ///
    /// Header row is printed once per application start
    /// (Image, PreprocessMs, InferenceMs, TotalMs, CpuMs, BeforeMiB, AfterMiB, ReductionPct).
    /// </summary>
    private static void LogMetricsCsv(
        double preMs, double infMs, double totalMs, double cpuMs,
        HsiCube source, HsiCube preprocessed)
    {
        if (!_headerPrinted)
        {
            Console.WriteLine("Image,PreprocessMs,InferenceMs,TotalMs,CpuMs,BeforeMiB,AfterMiB,ReductionPct");
            Debug.WriteLine("Image,PreprocessMs,InferenceMs,TotalMs,CpuMs,BeforeMiB,AfterMiB,ReductionPct");
            _headerPrinted = true;
        }

        var bytesPerElement = source.Header.DataType switch
        {
            1 => 1, // byte
            2 => 2, // int16
            3 => 4, // int32
            4 => 4, // float32
            5 => 8, // float64
            12 => 2, // uint16
            _ => 4, // unknown -> float32
        };

        var beforeMiB = (long)source.Samples * source.Lines * source.Bands * bytesPerElement / (1024.0 * 1024.0);
        var afterMiB = (long)preprocessed.Samples * preprocessed.Lines * preprocessed.Bands * bytesPerElement /
                       (1024.0 * 1024.0);
        var reduction = beforeMiB > 0 ? (1.0 - afterMiB / beforeMiB) * 100.0 : 0.0;

        var image = Path.GetFileName(Path.GetDirectoryName(source.Header.DataFilePath)) ?? "(unknown)";
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1:F0},{2:F0},{3:F0},{4:F0},{5:F2},{6:F2},{7:F2}",
            image, preMs, infMs, totalMs, cpuMs, beforeMiB, afterMiB, reduction));

        Debug.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1:F0},{2:F0},{3:F0},{4:F0},{5:F2},{6:F2},{7:F2}",
            image, preMs, infMs, totalMs, cpuMs, beforeMiB, afterMiB, reduction));
    }


    // -- Runs (saved classification results) ------------------------------------ /
    [ObservableProperty] [NotifyCanExecuteChangedFor(nameof(ExportPdfCommand))]
    private RunSummary? _activeRun;

    /// <summary>
    /// View-wrappers around <see cref="ImageNode"/>.Runs that carry per-card view-state
    /// (notably <see cref="RunCardViewModel.IsActive"/>) so XAML can bind to a simple bool.
    /// </summary>
    public ObservableCollection<RunCardViewModel> RunCards { get; } = [];

    public bool HasNoRuns => RunCards.Count == 0;

    private void RebuildRunCards()
    {
        RunCards.Clear();
        foreach (var run in ImageNode.Runs)
            RunCards.Add(new RunCardViewModel(run, run.RunId == ActiveRun?.RunId));
        OnPropertyChanged(nameof(HasNoRuns));
    }

    private void OnRunsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        => RebuildRunCards();

    partial void OnActiveRunChanged(RunSummary? value)
    {
        var activeId = value?.RunId;
        foreach (var card in RunCards)
            card.IsActive = card.RunId == activeId;
    }

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
            Debug.WriteLine("Auto Save Run failed: " + ex.Message);
            return null;
        }
    }

    [RelayCommand]
    private async Task LoadRun(RunSummary? summary)
    {
        if (summary == null || !InLibraryMode) return;

        if (Cube == null)
            return;

        var report = await _libraryManager.LoadRunAsync(ImageNode.ImageId, summary.RunId, _cts.Token);
        if (report == null)
            return;

        Overlay.ApplyResult(report, Cube.Samples, Cube.Lines);
        ActiveRun = summary;
    }

    /// <summary>
    /// Deletes a saved run from disk and the library manifest, and removes it from the
    /// runs list. If the deleted run was currently displayed, clears the overlay.
    /// </summary>
    [RelayCommand]
    private async Task DeleteRun(RunSummary? summary)
    {
        if (summary == null || !InLibraryMode) return;

        var confirmed = await _dialogService.ConfirmAsync(
            title: "Delete run",
            message:
            $"Delete the run from {summary.CompletedAt:yyyy-MM-dd HH:mm} ({summary.ModelDisplayName})? This cannot be undone.",
            confirmLabel: "Delete",
            isDestructive: true);
        if (!confirmed) return;

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
            Debug.WriteLine("Delete Run failed: " + ex.Message);
        }
    }


    // -- Notes (per-image clinical notes) ------------------------ //

    [ObservableProperty] private bool _showNotes;

    /// <summary>
    /// Clinical notes attached to the image itself (not to a specific run).
    /// Persisted as <see cref="ImageNode.Notes"/> via <see cref="LibraryManager"/>.
    /// </summary>
    [ObservableProperty] private string _imageNotes = "";

    private bool CanSaveNotes() => InLibraryMode;

    [RelayCommand]
    private void ToggleNotes() => ShowNotes = !ShowNotes;

    /// <summary>
    /// Persists <see cref="ImageNotes"/> to the image's manifest entry.
    /// Notes are per-image: they describe the sample/tissue itself,
    /// independent of which model produced which run.
    /// </summary>
    [RelayCommand(CanExecute = nameof(CanSaveNotes))]
    private async Task SaveNotes(CancellationToken ct)
    {
        if (!InLibraryMode) return;
        try
        {
            await _libraryManager.UpdateImageNotesAsync(ImageNode.ImageId, ImageNotes, ct);
        }
        catch (Exception ex)
        {
            Debug.WriteLine("Failed to save notes: " + ex.Message);
        }
    }


    // -- Spectral signature (pixel picks) ----------------------------- /
    
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


    // -- PDF export ----------------------------------------------------------- /

    private bool CanExportPdf() => Cube != null && Overlay.ClassificationResult != null;

    [RelayCommand(CanExecute = nameof(CanExportPdf))]
    private async Task ExportPdfAsync()
    {
        if (Cube == null || Overlay.ClassificationResult == null || Overlay.CachedHeatmap == null) return;

        var ownerWindow = Application.Current?.MainWindow();
        if (ownerWindow == null) return;

        var defaults = ExportOptions.FromOverlay(
            Overlay.SelectedColorMapName, (float)Overlay.OverlayOpacity, (float)Overlay.OverlayThreshold);

        var options = await _dialogService.ShowExportDialogAsync(defaults);
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
                PdfReportExporter.Export(report, rgb, heatmap, options, ms, ImageNode.CurrentRelPath, ImageNotes);
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
        _session = new SessionService();
        _dialogService = null!;

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
            new PixelSize(320, 240),
            new Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Opaque);

        LoadingState = LoadingState.Ready;
        StatusMessage = "Design preview";
    }
}