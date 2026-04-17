using System.Collections.Generic;
using System.Linq;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Rendering;

namespace SpectralAssist.ViewModels;

/// <summary>
/// Manages the overlay, colormap, threshold and info panel states for the classification overlay.
/// </summary>
public partial class OverlayManager : ObservableObject
{
    private HsiCube? _cube;

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
    /// Applies a new classification result and rebuilds the overlay bitmaps.
    /// Called by the ViewModel after inference completes.
    /// </summary>
    public void ApplyResult(ClassificationResult result, HsiCube cube)
    {
        _cube = cube;
        ClassificationResult = result;
        // Ensure a real colormap is selected (not "Off") when inference completes
        if (SelectedColorMapName == "Off")
            SelectedColorMapName = "Green-Red";
        ShowOverlay = true;
        RebuildOverlay();
    }

    /// <summary>
    /// Rebuilds the overlay and color bar bitmaps from current state.
    /// Called automatically when colormap or threshold changes.
    /// </summary>
    private void RebuildOverlay()
    {
        if (ClassificationResult == null || _cube == null) return;

        var colorMap = ColorMaps.All.GetValueOrDefault(SelectedColorMapName, ColorMaps.GreenRed);
        OverlayBitmap = BitmapRenderer.ClassificationOverlay(
            _cube,
            ClassificationResult,
            colorMap,
            threshold: (float)OverlayThreshold);
        ColorBarBitmap = BitmapRenderer.ColorBarLegend(
            colorMap,
            threshold: (float)OverlayThreshold);
    }

    /// <summary>
    /// Clears all overlay state. Called on dispose or image reload.
    /// </summary>
    public void Clear()
    {
        _cube = null;
        ClassificationResult = null;
        OverlayBitmap = null;
        ColorBarBitmap = null;
    }
}