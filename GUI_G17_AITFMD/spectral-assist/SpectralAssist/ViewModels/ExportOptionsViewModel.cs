using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

public partial class ExportOptionsViewModel : ViewModelBase
{
    [ObservableProperty] private double _overlay1Threshold = 0.5;
    [ObservableProperty] private double _overlay2Threshold = 0.8;
    [ObservableProperty] private string _selectedColorMapName = "Green-Red";
    [ObservableProperty] private double _opacity = 0.5;

    public IReadOnlyList<string> AvailableColorMaps { get; } =
        ColorMaps.All.Keys.ToList();

    public ExportOptions BuildOptions() =>
        new(
            Overlay1Threshold: (float)Overlay1Threshold,
            Overlay2Threshold: (float)Overlay2Threshold,
            ColorMapName: SelectedColorMapName,
            Opacity: (float)Opacity);
}
