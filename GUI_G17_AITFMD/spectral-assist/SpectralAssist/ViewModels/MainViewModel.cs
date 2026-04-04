using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel(ImageLoadingService loadingService, InferenceService inference) : ViewModelBase
{
    // -- ViewModels -- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasImageView))]
    [NotifyCanExecuteChangedFor(nameof(RunInferenceCommand))]
    private ViewModelBase _currentView = new HomeViewModel();
    
    public bool HasImageView => CurrentView is ImageViewModel;

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

    // -- Navigation -- //
    [RelayCommand]
    private void NavigateToHome()
    {
        DisposeCurrentView();
        CurrentView = new HomeViewModel();
    }

    [RelayCommand]
    public void NavigateToImage(string filePath)
    {
        DisposeCurrentView();
        CurrentView = new ImageViewModel(filePath, loadingService, inference);
    }

    // -- Inference Action hoisted from ImageViewModel -- //
    [RelayCommand(CanExecute = nameof(HasImageView))]
    private void RunInference()
    {
        if (CurrentView is ImageViewModel imageVm)
            imageVm.RunInferenceCommand.Execute(null);
    }

    private void DisposeCurrentView()
    {
        if (CurrentView is IDisposable disposable)
            disposable.Dispose();
    }
}