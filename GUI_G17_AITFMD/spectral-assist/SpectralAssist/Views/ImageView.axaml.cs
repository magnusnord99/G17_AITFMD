using System;
using Avalonia.Controls;
using Avalonia.Input;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class ImageView : UserControl
{
    private ImageViewModel Vm => (ImageViewModel)DataContext!;

    public ImageView()
    {
        InitializeComponent();
    }

    private void ImageContainer_OnPointerPressed(object? sender, PointerPressedEventArgs args)
    {
        if (!Vm.IsPixelSpectrumMode || Vm.Cube == null) return;

        if (this.FindControl<Image>("DisplayImage") is not { } image) return;
        if (image.Bounds.Width <= 0 || image.Bounds.Height <= 0) return;

        var point = args.GetPosition(image);
        if (point.X < 0 || point.Y < 0 || point.X > image.Bounds.Width || point.Y > image.Bounds.Height)
            return;

        var cube = Vm.Cube;
        var x = (int)Math.Clamp(point.X * cube.Samples / image.Bounds.Width, 0, cube.Samples - 1);
        var y = (int)Math.Clamp(point.Y * cube.Lines / image.Bounds.Height, 0, cube.Lines - 1);

        Vm.SelectPixelSpectrumAt(x, y);
    }
}