using System.Collections.Generic;
using System.Linq;
using Avalonia.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

public partial class ExportDialogViewModel : ViewModelBase
{
    public ExportDialogViewModel(Window dialog, ExportOptions? initial = null)
    {
        _dialog = dialog;

        var defaults = initial ?? ExportOptions.Default;
        _selectedColorMapName = defaults.ColorMapName;
        _opacity = defaults.OverlayOpacity;
        _overlay1Threshold = defaults.Overlay1Threshold;
        _overlay2Threshold = defaults.Overlay2Threshold;
    }

    private readonly Window _dialog;
    [ObservableProperty] private string _selectedColorMapName;
    [ObservableProperty] private double _opacity;
    [ObservableProperty] private double _overlay1Threshold;
    [ObservableProperty] private double _overlay2Threshold;

    public IReadOnlyList<string> AvailableColorMaps { get; } = ColorMaps.All.Keys.ToList();

    private ExportOptions BuildOptions() => new(
        Overlay1Threshold: (float)Overlay1Threshold,
        Overlay2Threshold: (float)Overlay2Threshold,
        ColorMapName: SelectedColorMapName,
        OverlayOpacity: (float)Opacity);

    [RelayCommand]
    private void Confirm() => _dialog.Close(BuildOptions());

    [RelayCommand]
    private void Cancel() => _dialog.Close(null);
}