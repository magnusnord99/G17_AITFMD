using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly ModelPackageManager _modelManager;
    private readonly LibraryViewModel _libraryView;
    private readonly ModelsViewModel _modelsView;
    private readonly Func<ImageNode, ImageViewModel> _imageVmFactory;
    
    public MainViewModel(
        ModelPackageManager modelManager,
        SessionService session,
        LibraryViewModel libraryView,
        ModelsViewModel modelsView,
        Func<ImageNode, ImageViewModel> imageVmFactory)
    {
        _modelManager = modelManager;
        Session = session;
        _libraryView = libraryView;
        _modelsView = modelsView;
        _imageVmFactory = imageVmFactory;
        
        _currentView = libraryView;
        _libraryView.ImageSelected += OpenImageFromLibrary;
    }

    // --- Observable States --- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasOpenImage))]
    [NotifyPropertyChangedFor(nameof(OpenImageDisplayName))]
    private ImageViewModel? _openImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsOnImageView))]
    [NotifyPropertyChangedFor(nameof(IsOnLibrary))]
    [NotifyPropertyChangedFor(nameof(IsOnModels))]
    private ViewModelBase _currentView;

    public bool HasOpenImage => OpenImage != null;
    public string OpenImageDisplayName => OpenImage?.ImageNode.CurrentRelPath ?? "";

    public bool IsOnImageView => CurrentView is ImageViewModel;
    public bool IsOnLibrary => CurrentView is LibraryViewModel;
    public bool IsOnModels => CurrentView is ModelsViewModel;

    public SessionService Session { get; }
    public ObservableCollection<ModelManifest> AvailableModels => _modelManager.AvailableModels;

    // --- Navigation --- //
    [RelayCommand]
    private void NavigateToLibrary() => CurrentView = _libraryView;

    [RelayCommand]
    private void NavigateToModels() => CurrentView = _modelsView;
    
    [RelayCommand]
    private void NavigateToOpenImage()
    {
        if (OpenImage != null) CurrentView = OpenImage;
    }

    [RelayCommand]
    public void OpenImageFromPath(string filePath)
    {
        // If chosen image is already open, just switch to it.
        if (OpenImage is { } current
            && string.Equals(current.ImageNode.AbsolutePath, filePath, StringComparison.OrdinalIgnoreCase))
        {
            CurrentView = OpenImage;
            return;
        }

        OpenImage?.Dispose();
        var transientNode = ImageNode.CreateTransient(filePath);
        OpenImage = _imageVmFactory(transientNode);
        //OpenImage.CloseRequested += CloseOpenImage;  // ToDO: Is this needed?
        Session.ActiveImageId = transientNode.ImageId;
        CurrentView = OpenImage;
    }
    
    [RelayCommand]
    private void CloseOpenImage()
    {
        OpenImage?.Dispose();
        OpenImage = null;
        Session.ActiveImageId = null;
        NavigateToLibrary();
    }

    private void OpenImageFromLibrary(ImageNode imageNode)
    {
        // If the clicked image is already open, just switch to it.
        if (OpenImage is { } current && current.ImageNode.ImageId == imageNode.ImageId)
        {
            CurrentView = OpenImage;
            return;
        }

        OpenImage?.Dispose();
        OpenImage = _imageVmFactory(imageNode);
        //OpenImage.CloseRequested += CloseOpenImage;  // ToDO: Is this needed?
        Session.ActiveImageId = imageNode.ImageId;
        CurrentView = OpenImage;
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
        _modelManager = new ModelPackageManager();
        Session = new SessionService();
        _libraryView = null!;
        _modelsView = null!;
        _currentView = null!;
        _imageVmFactory = (_) => null!;
    }
}