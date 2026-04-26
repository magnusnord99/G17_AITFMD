using System.Collections.Generic;
using System.Linq;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels.Components;

/// <summary>
/// Manages the overlay, colormap and thresholding states for the classification overlay.
/// </summary>
public partial class OverlayViewModel : ObservableObject
{
    private float[]? _cachedHeatmap;
    private int _heatmapWidth;
    private int _heatmapHeight;

    // -- States -- //
    [ObservableProperty] private Bitmap? _overlayBitmap;
    [ObservableProperty] private Bitmap? _colorBarBitmap;
    [ObservableProperty] private double _overlayOpacity = 0.5;
    [ObservableProperty] private bool _showOverlay = true;
    [ObservableProperty] private double _overlayThreshold;
    [ObservableProperty] private ClassificationReport? _classificationResult;
    [ObservableProperty] private string _selectedColorMapName = ColorMaps.All.Keys.First();
    
    public IReadOnlyList<string> AvailableColorMaps { get; } = ColorMaps.All.Keys.ToList();
    partial void OnSelectedColorMapNameChanged(string value) => RebuildOverlay();
    partial void OnOverlayThresholdChanged(double value) => RebuildOverlay();
    
    /// <summary>
    /// Applies a new classification result and performs one-time build of the Gaussian-weighted heatmap,
    /// then renders the overlay bitmap.
    /// </summary>
    public void ApplyResult(ClassificationReport report, int imageWidth, int imageHeight)
    {
        ClassificationResult = report;
        _heatmapWidth = imageWidth;
        _heatmapHeight = imageHeight;
        _cachedHeatmap = HeatmapRenderer.BuildHeatmap(report, _heatmapWidth, _heatmapHeight);
        
        ShowOverlay = true;
        RebuildOverlay();
    }

    /// <summary>
    /// Re-renders the overlay bitmap from the cached heatmap using current colormap and threshold.
    /// Called automatically when colormap or threshold changes.
    /// </summary>
    private void RebuildOverlay()
    {
        if (_cachedHeatmap == null || ClassificationResult == null) return;

        var colorMap = ColorMaps.All.GetValueOrDefault(SelectedColorMapName, ColorMaps.GreenRed);
        OverlayBitmap = HeatmapRenderer.RenderHeatmap(
            _cachedHeatmap,
            _heatmapWidth,
            _heatmapHeight,
            colorMap,
            threshold: (float)OverlayThreshold);
        ColorBarBitmap = HeatmapRenderer.ColorBarLegend(
            colorMap,
            threshold: (float)OverlayThreshold);
    }

    /// <summary>
    /// Clears all overlay state. Called on dispose or image reload.
    /// </summary>
    public void Clear()
    {
        _cachedHeatmap = null;
        ClassificationResult = null;
        OverlayBitmap = null;
        ColorBarBitmap = null;
    }
}