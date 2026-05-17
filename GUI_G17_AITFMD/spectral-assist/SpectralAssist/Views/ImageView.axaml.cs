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
        SpectrumPlot.Plot.SetStyle(new ScottPlot.PlotStyles.Dark());
        SpectrumPlot.Plot.FigureBackground.Color = ScottPlot.Colors.Transparent;
        SpectrumPlot.UserInputProcessor.Disable();
    }

    private void Sync(ZoomBorder target, ZoomChangedEventArgs e)
    {
        if (_syncing) return;
        _syncing = true;
        try
        {
            target.SetMatrix(new Matrix(e.ZoomX, 0, 0, e.ZoomY, e.OffsetX, e.OffsetY));
        }
        finally
        {
            _syncing = false;
        }
    }

    private void OnImagePointerPressed(object? sender, PointerPressedEventArgs e)
    {
        var props = e.GetCurrentPoint((Image)sender!).Properties;
        if (!props.IsLeftButtonPressed) return;

        var hasShift = e.KeyModifiers.HasFlag(KeyModifiers.Shift);
        var hasCtrl = e.KeyModifiers.HasFlag(KeyModifiers.Control);
        if (!hasShift && !hasCtrl) return;

        var img = (Image)sender!;
        var pos = e.GetPosition(img);
        var x = (int)pos.X;
        var y = (int)pos.Y;

        if (DataContext is not ImageViewModel vm) return;

        vm.OnPixelClicked(x, y, hasShift);
        UpdateSpectrumPlot(vm);
        e.Handled = true;
    }

    private void UpdateSpectrumPlot(ImageViewModel vm)
    {
        if (vm.Cube == null) return;
        var wavelengths = vm.Cube.Header.WavelengthValues;
        SpectrumPlot.Plot.Clear();
        
        if (vm is { Pixel1X: { } p1X, Pixel1Y: { } p1Y })
        {
            var spectrum = vm.Cube.GetSpectrumAt(p1X, p1Y);
            var yellowScatter = SpectrumPlot.Plot.Add.ScatterLine(wavelengths, spectrum);
            yellowScatter.LineColor = ScottPlot.Colors.Yellow;
            yellowScatter.LineWidth = 1.5f;
        }
        if (vm is { Pixel2X: { } p2X, Pixel2Y: { } p2Y })
        {
            var spectrum = vm.Cube.GetSpectrumAt(p2X, p2Y);
            var redScatter = SpectrumPlot.Plot.Add.ScatterLine(wavelengths, spectrum);
            redScatter.LineColor = ScottPlot.Colors.Red;
            redScatter.LineWidth = 1.5f;
        }
        
        SpectrumPlot.Plot.Axes.SetLimitsY(0, 1.1);
        SpectrumPlot.Plot.Axes.SetLimitsX(wavelengths.Min(), wavelengths.Max()); 
        SpectrumPlot.Refresh();
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        if (e.Key == Key.Escape && DataContext is ImageViewModel vm)
        {
            vm.ClearPixelSelections();
            SpectrumPlot.Plot.Clear();
            SpectrumPlot.Refresh();
            e.Handled = true;
        }

        base.OnKeyDown(e);
    }

    private void OnClearSpectrumClicked(object? sender, RoutedEventArgs e)
    {
        if (DataContext is ImageViewModel vm)
        {
            vm.ClearPixelSelections();
            SpectrumPlot.Plot.Clear();
            SpectrumPlot.Refresh();
        }
    }
}