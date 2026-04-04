using Avalonia.Controls;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class HomeView : UserControl
{
    private HomeViewModel Vm => (HomeViewModel)DataContext!;
    
    public HomeView()
    {
        InitializeComponent();
    }
    
}