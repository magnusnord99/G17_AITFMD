using CommunityToolkit.Mvvm.ComponentModel;

namespace SpectralAssist.ViewModels;

public partial class HomeViewModel : ViewModelBase
{
    [ObservableProperty] private string _title = "This is the home page";
}