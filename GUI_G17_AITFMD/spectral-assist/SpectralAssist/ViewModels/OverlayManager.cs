using System.Collections.Generic;
using System.Linq;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

/// <summary>
/// Manages the overlay, colormap, threshold and info panel states for the classification overlay.
/// </summary>
public partial class OverlayManager : ObservableObject
{
    private float[]? _cachedHeatmap;
    private int _heatmapWidth;
    private int _heatmapHeight;

    // -- States -- //
    [ObservableProperty] private WriteableBitmap? _overlayBitmap;
    [ObservableProperty] private WriteableBitmap? _colorBarBitmap;
    [ObservableProperty] private double _overlayOpacity = 0.5;
    [ObservableProperty] private bool _showOverlay = true;
    [ObservableProperty] private double _overlayThreshold;
    [ObservableProperty] private ClassificationResult? _classificationResult;
    [ObservableProperty] private bool _showInfoPanel;
    [ObservableProperty] private string _selectedColorMapName = "Green-Red";

    /// <summary>Available colormap names for the dropdown (includes "Off" to hide overlay).</summary>
    public IReadOnlyList<string> AvailableColorMaps { get; } =
        new[] { "Off" }.Concat(ColorMaps.All.Keys).ToList();

    // -- Property change handlers -- //
    partial void OnSelectedColorMapNameChanged(string value)
    {
        ShowOverlay = value != "Off";
        if (ShowOverlay) RebuildOverlay();
    }

    partial void OnOverlayThresholdChanged(double value) => RebuildOverlay();

    
    /// <summary>
    /// Applies a new classification result and performs one-time build of the Gaussian-weighted heatmap,
    /// then renders the overlay bitmap.
    /// </summary>
    public void ApplyResult(ClassificationResult result, int imageWidth, int imageHeight)
    {
        // Build the heatmap once
        ClassificationResult = result;
        _heatmapWidth = imageWidth;
        _heatmapHeight = imageHeight;
        _cachedHeatmap = HeatmapRenderer.BuildHeatmap(result, _heatmapWidth, _heatmapHeight);
        
        if (SelectedColorMapName == "Off")
            SelectedColorMapName = "Green-Red";
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