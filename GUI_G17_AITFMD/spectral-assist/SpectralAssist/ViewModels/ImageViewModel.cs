using System;
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
    private readonly string  _hdrPath;

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
    [NotifyPropertyChangedFor(nameof(SelectedBandLabel))]
    private HsiCube? _cube;
    
    [ObservableProperty] private DisplayMode _currentDisplayMode = DisplayMode.Rgb;
    
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedBandLabel))]
    private int _selectedBand;
    
    [ObservableProperty] private string _statusMessage = "";
    [ObservableProperty] private double _progress;
    [ObservableProperty] private WriteableBitmap? _currentBitmap;
    [ObservableProperty] private string _inferenceOutput = "";
    
    public bool IsLoading => LoadingState == LoadingState.Loading;
    public bool IsError => LoadingState == LoadingState.Error;
    public bool IsReady => LoadingState == LoadingState.Ready;
    public int MaxBandIndex => Cube?.Bands - 1 ?? 0;
    public string SelectedBandLabel => Cube?.Header.WavelengthUnit ?? "Unit";
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

    public IAsyncRelayCommand RunInferenceCommand { get; }

    private async Task RunInferenceAsync()
    {
        InferenceOutput = "Running...";
        var (ok, output) = await Task.Run(() => InferenceRunner.Run(_hdrPath));
        InferenceOutput = ok ? output : $"Error: {output}";
    }
    
    
    private async Task LoadAsync()
    {
        try
        {
            LoadingState = LoadingState.Loading;
            StatusMessage = "Reading header...";
            Progress = 0;
            
            // Step 1: instant - parse header and show info
            var header = HsiHeaderParser.Parse(_hdrPath);
            StatusMessage = $"{header.Samples}x{header.Lines}x{header.Bands} bands, {header.Interleave.ToUpper()}";
            
            
            // Step 2: heavy load - load and convert binary data into memory
            var progressReporter = new Progress<(float percent, int band)>(p =>
            {
                Progress = p.percent;
                StatusMessage = $"Loading image data... {p.percent:P0}";
            });
            Cube = await HsiCubeLoader.LoadAsync(header, progressReporter, _cts.Token);
            
            // Step 3: Show image or something
            LoadingState = LoadingState.Ready;
            StatusMessage = "Loading Complete";
            UpdateBitmap();
        }
        
        catch (OperationCanceledException)
        {
            LoadingState = LoadingState.Idle;
            StatusMessage = "Operation Canceled";;
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
                Cube.Header.DefaultBands[0],
                Cube.Header.DefaultBands[1],
                Cube.Header.DefaultBands[2]),
            _ => CurrentBitmap
                  
        };
    }
    
    public void Dispose()
    {
        Console.WriteLine("ImageViewModel.Dispose() called");
        _cts.Cancel();
        _cts.Dispose();
        CurrentBitmap = null;
        Cube = null;
        GC.SuppressFinalize(this);
    }
}
