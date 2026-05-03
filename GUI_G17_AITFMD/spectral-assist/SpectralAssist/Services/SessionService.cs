using System.Collections.Generic;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

public partial class SessionService : ObservableObject
{
    // Inference preferences
    [ObservableProperty] private ModelManifest? _activeModel;
    [ObservableProperty] private StrideOption _selectedStride = StrideOption.Default;
    public static IReadOnlyList<StrideOption> AvailableStrides => StrideOption.Presets;
    
    // Library selection
    [ObservableProperty] private string? _activeImageId;
}