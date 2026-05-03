using System;
using System.Collections.ObjectModel;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Library;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly LibraryManager _libraryManager;
    private readonly ModelPackageManager _modelManager;
    private readonly LibraryViewModel _libraryView;
    private readonly ModelsViewModel _modelsView;
    private readonly Func<string, string?, ImageViewModel> _imageVmFactory;
    private ImageViewModel? _imageView;

    public MainViewModel(
        LibraryManager libraryManager,
        ModelPackageManager modelManager,
        SessionService session,
        LibraryViewModel libraryView,
        ModelsViewModel modelsView,
        Func<string, string?, ImageViewModel> imageVmFactory)
    {
        _libraryManager = libraryManager;
        _modelManager = modelManager;
        Session = session;
        _libraryView = libraryView;
        _modelsView = modelsView;
        _imageVmFactory = imageVmFactory;

        _libraryView.ImageSelected += OpenImageFromLibrary;
    }
    
    // --- Observable States --- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasImageView))]
    [NotifyPropertyChangedFor(nameof(IsOnImageView))]
    private ViewModelBase _currentView = new HomeViewModel();
    
    public SessionService Session { get; }
    public ObservableCollection<ModelManifest> AvailableModels => _modelManager.AvailableModels;
    public bool HasImageView => _imageView != null;
    public bool IsOnImageView => CurrentView is ImageViewModel;
    
    // --- Navigation --- //
    [RelayCommand]
    private void NavigateToHome() => CurrentView = (ViewModelBase?)_imageView ?? new HomeViewModel();

    [RelayCommand]
    private void NavigateToModels() => CurrentView = _modelsView;

    [RelayCommand]
    private void NavigateToLibrary() =>CurrentView = _libraryView;
    
    [RelayCommand]
    public void OpenImage(string filePath)
    {
        _imageView?.Dispose();
        _imageView = _imageVmFactory(filePath, null);
        Session.ActiveImageId = null;
        CurrentView = _imageView;
    }

    private void OpenImageFromLibrary(ImageNode imageNode)
    {
        if (_libraryManager.Root == null) return;

        var absPath = Path.Combine(
            _libraryManager.Root,
            imageNode.CurrentRelPath.Replace('/', Path.DirectorySeparatorChar));

        _imageView?.Dispose();
        _imageView = _imageVmFactory(absPath, imageNode.ImageId);
        Session.ActiveImageId = imageNode.ImageId;
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
        _libraryManager = null!;
        _modelManager = new ModelPackageManager();
        Session = new SessionService();
        _libraryView = null!;
        _modelsView = null!;
        _imageVmFactory = (_, _) => null!;
    }
}