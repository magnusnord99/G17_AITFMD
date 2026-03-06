using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace SpectralAssist.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    // --- ViewModels --- //
    [ObservableProperty] private ViewModelBase _currentView = new HomeViewModel();
    
    // --- Drag and Drop Functionality --- //
    [ObservableProperty] private bool _isDragging;
    [ObservableProperty] private string _dragIcon = "⬇";
    [ObservableProperty] private string _dragMessage = "Drop to open HSI file";
    
    public void SetDragState(bool dragging, bool valid)
    {
        IsDragging = dragging;
        DragIcon = valid ? "⬇" : "✕";
        DragMessage = valid ? "Drop to open HSI file" : "Unsupported: please use a .hdr file";
    }
    
    // --- Navigation --- //
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
        CurrentView = new ImageViewModel(filePath);
    }
    
    private void DisposeCurrentView()
    {
        if (CurrentView is not IDisposable disposable) return;
        
        Console.WriteLine($"Disposing current {CurrentView?.GetType().Name}");
        disposable.Dispose();
        
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}