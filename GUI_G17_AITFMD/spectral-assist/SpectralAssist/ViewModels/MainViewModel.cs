using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Library;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly InferenceService _inferenceService;
    private readonly LibraryManager _libraryManager;
    private readonly ModelPackageManager _modelManager;
    
    private ImageViewModel? _imageView;
    private LibraryViewModel? _libraryView;
    
    public MainViewModel(
        InferenceService inferenceService,
        LibraryManager libraryManager,
        ModelPackageManager modelManager)
    {
        _inferenceService = inferenceService;
        _modelManager = modelManager;
        _libraryManager = libraryManager;
        _modelManager.Refresh();
        
        // ToDo: Change this from FirstOrDefault to settings based preferred model or last used with persistence?
        ActiveModel = _modelManager.AvailableModels.FirstOrDefault();
        _selectedStride = AvailableStrides[0];
    }

    // --- Observable States --- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasImageView))]
    [NotifyPropertyChangedFor(nameof(IsOnImageView))]
    [NotifyCanExecuteChangedFor(nameof(RunInferenceCommand))]
    private ViewModelBase _currentView = new HomeViewModel();
    
    [ObservableProperty] private ModelManifest? _activeModel;
    [ObservableProperty] private StrideOption _selectedStride = StrideOption.Default;
    
    public static IReadOnlyList<StrideOption> AvailableStrides => StrideOption.Presets;
    public bool HasImageView => _imageView != null;
    public bool IsOnImageView => CurrentView is ImageViewModel;
    
    // --- Commands --- //
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
        CurrentView = new ModelsViewModel(_modelManager, _inferenceService, ActiveModel, 
            modelManifest => ActiveModel = modelManifest);
    }
    
    [RelayCommand]
    private void NavigateToLibrary()
    {
        _libraryView ??= new LibraryViewModel(_libraryManager, OpenImageFromLibrary);
        _libraryView.RefreshView();
        CurrentView = _libraryView;
    }
    
    [RelayCommand]
    public void OpenImage(string filePath)
    {
        // Dispose the previous image (if any) then load new one
        _imageView?.Dispose();
        _imageView = new ImageViewModel(filePath, _inferenceService);
        _libraryView?.ActiveImageId = null;
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
        var modelPackage = _modelManager.LoadPackage(selected.DirectoryPath);
        var spec = modelPackage.Manifest.InputSpec;
        var patchSize = spec.SpatialPatchSize[0];
        var stride = SelectedStride.Divisor switch
        {
            0  => spec.Stride.FirstOrDefault(patchSize),
            var divisor => patchSize / divisor,
        };
        
        await _imageView.RunInference(modelPackage, stride);
    }
    
    private void OpenImageFromLibrary(ImageNode imageNode)
    {
        if (_libraryManager.Root == null) return;

        var absPath = Path.Combine(_libraryManager.Root, imageNode.CurrentRelPath.Replace('/', Path.DirectorySeparatorChar));
        _imageView?.Dispose();
        _imageView = new ImageViewModel(absPath, _inferenceService, imageNode.ImageId, _libraryManager);
        _libraryView?.ActiveImageId = imageNode.ImageId;
        CurrentView = _imageView;
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
        _inferenceService = null!;
        _libraryManager = null!;
        _modelManager = new ModelPackageManager();
    }
}