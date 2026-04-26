using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.PanAndZoom;
using Avalonia.Input;
using Avalonia.Interactivity;
using SpectralAssist.ViewModels;
using Image = Avalonia.Controls.Image;

namespace SpectralAssist.Views;

public partial class ImageView : UserControl
{
    private ImageViewModel Vm => (ImageViewModel)DataContext!;
    private bool _syncing;
    
    
    public ImageView()
    {
        InitializeComponent();
        ZoomBorderLeft.ZoomChanged += (_, e) => Sync(ZoomBorderRight, e);
        ZoomBorderRight.ZoomChanged += (_, e) => Sync(ZoomBorderLeft, e);
    }
    
    private void Sync(ZoomBorder target, ZoomChangedEventArgs e)
    {
        if (_syncing) return;
        _syncing = true;
        try
        {
            target.SetMatrix(new Matrix(e.ZoomX, 0, 0, e.ZoomY, e.OffsetX, e.OffsetY));
        }
        finally { _syncing = false; }
    }
    
    private void OnImagePointerPressed(object? sender, PointerPressedEventArgs e)
    {
        var props = e.GetCurrentPoint((Image)sender!).Properties;
        if (!props.IsLeftButtonPressed) return;
        if (!e.KeyModifiers.HasFlag(KeyModifiers.Shift)) return;

        var img = (Image)sender!;
        var pos = e.GetPosition(img);
        var x = (int)pos.X;
        var y = (int)pos.Y;

        if (DataContext is not ImageViewModel vm) return;

        vm.OnPixelClicked(x, y);
        UpdateSpectrumPlot(vm, x, y);
        e.Handled = true;
    }

    private void UpdateSpectrumPlot(ImageViewModel vm, int x, int y)
    {
        if (vm.Cube == null) return;

        var spectrum = vm.Cube.GetSpectrumAt(x, y);
        var wavelengths = vm.Cube.Header.WavelengthValues;

        SpectrumPlot.Plot.Clear();
        SpectrumPlot.Plot.Add.ScatterLine(wavelengths, spectrum);
        SpectrumPlot.Plot.XLabel($"Wavelength ({vm.WavelengthUnit})");
        SpectrumPlot.Plot.YLabel("Reflectance");
        SpectrumPlot.Plot.Title($"Pixel ({x}, {y})");
        SpectrumPlot.Plot.Axes.SetLimitsY(0, 1.1); 
        SpectrumPlot.Plot.Axes.SetLimitsX(wavelengths.Min(), wavelengths.Max()); 
        SpectrumPlot.Plot.SetStyle(new ScottPlot.PlotStyles.Dark());
        SpectrumPlot.Plot.Add.Palette = new ScottPlot.Palettes.Aurora();
        SpectrumPlot.Refresh();
    }
    
    private void OnClearSpectrumClicked(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ImageViewModel vm)
        {
            vm.SelectedX = null;
            vm.SelectedY = null;
            SpectrumPlot.Plot.Clear();
            SpectrumPlot.Refresh();
        }
    }
}