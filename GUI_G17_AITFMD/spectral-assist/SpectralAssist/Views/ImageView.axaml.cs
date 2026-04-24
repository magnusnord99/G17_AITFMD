using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.PanAndZoom;
using Avalonia.Input;
using SpectralAssist.ViewModels;

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
        try
        {
            var props = e.GetCurrentPoint(null).Properties;
            if (!props.IsLeftButtonPressed) return;
            if (e.KeyModifiers != KeyModifiers.Shift) return;
            
            var img = (Image)sender!;
            var pos = e.GetPosition(img);
            var x = (int)pos.X;
            var y = (int)pos.Y;
            Console.WriteLine($"x = {x}, y = {y}");
            if (DataContext is ImageViewModel vm) 
                vm.OnPixelClicked(x, y);
        }
        catch (Exception exception)
        {
            Console.WriteLine(exception);
            Console.WriteLine(exception.Message);
        }
    }
}