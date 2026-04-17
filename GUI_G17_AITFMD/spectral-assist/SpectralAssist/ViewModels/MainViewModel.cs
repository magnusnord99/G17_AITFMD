using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly ImageLoadingService _loadingService;
    private readonly InferenceService _inferenceService;
    private ImageViewModel? _imageView;
    private readonly ModelPackageService _modelRegistry;
    
    public MainViewModel(
        ImageLoadingService loadingService,
        InferenceService inferenceService,
        ModelPackageService modelRegistry)
    {
        _loadingService = loadingService;
        _inferenceService = inferenceService;
        _modelRegistry = modelRegistry;
        _modelRegistry.Refresh();
        
        // ToDo: Change this from FirstOrDefault to settings based preferred model or last used with persistence?
        ActiveModel = _modelRegistry.AvailableModels.FirstOrDefault();
    }

    // -- Observables -- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasImageView))]
    [NotifyPropertyChangedFor(nameof(IsOnImageView))]
    [NotifyCanExecuteChangedFor(nameof(RunInferenceCommand))]
    private ViewModelBase _currentView = new HomeViewModel();
    
    [ObservableProperty] private ModelManifest? _activeModel;
    
    public bool HasImageView => _imageView != null;
    public bool IsOnImageView => CurrentView is ImageViewModel;


    // -- Navigation -- //
    [RelayCommand]
    private void NavigateToHome()
    {
        // If an image is loaded, go back to it instead of an empty home page
        if (_imageView != null)
            CurrentView = _imageView;
        else
            CurrentView = new HomeViewModel();
    }
    
    [RelayCommand]
    private void NavigateToModels()
    {
        CurrentView = new ModelsViewModel(_modelRegistry, _inferenceService, ActiveModel, modelManifest => ActiveModel = modelManifest);
    }
    
    
    // -- Actions -- //
    
    [RelayCommand]
    public void OpenImage(string filePath)
    {
        // Dispose the previous image (if any) then load new one
        _imageView?.Dispose();
        _imageView = new ImageViewModel(filePath, _loadingService, _inferenceService);
        CurrentView = _imageView;
    }
    
    [RelayCommand(CanExecute = nameof(HasImageView))]
    private async Task RunInference()
    {
        if (_imageView == null) return;

        var selected = ActiveModel;
        if (selected == null)
        {
            _imageView.InferenceOutput = "No model available. Import one via the Models page.";
            return;
        }

        CurrentView = _imageView;
        await _imageView.RunInference(selected.DirectoryPath);
    }
    
    
    
    // -- Drag and Drop Functionality -- //
    [ObservableProperty] private bool _isDragging;
    [ObservableProperty] private string _dragIcon = "⬇";
    [ObservableProperty] private string _dragMessage = "Drop to open HSI file";

    public void SetDragState(bool dragging, bool valid)
    {
        IsDragging = dragging;
        DragIcon = valid ? "⬇" : "✕";
        DragMessage = valid ? "Drop to open HSI file" : "Unsupported: please use a .hdr file";
    }

    
    
    /// <summary>Design preview constructor filled with dummy data.</summary>
    public MainViewModel()
    {
        _loadingService = null!;
        _inferenceService = null!;
        _modelRegistry = new ModelPackageService();
    }
}