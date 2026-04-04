using Avalonia.Controls;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class ImageView : UserControl
{
    private ImageViewModel Vm => (ImageViewModel)DataContext!;
    
    public ImageView()
    {
        InitializeComponent();
    }
}