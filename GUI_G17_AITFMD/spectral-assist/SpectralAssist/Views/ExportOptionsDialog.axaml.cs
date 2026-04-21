using Avalonia.Controls;
using Avalonia.Interactivity;
using SpectralAssist.Models;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class ExportOptionsDialog : Window
{
    public ExportOptions? Result { get; private set; }

    public ExportOptionsDialog()
    {
        InitializeComponent();
        DataContext = new ExportOptionsViewModel();
        CancelButton.Click += OnCancel;
        ConfirmButton.Click += OnConfirm;
    }

    private void OnConfirm(object? sender, RoutedEventArgs e)
    {
        Result = ((ExportOptionsViewModel)DataContext!).BuildOptions();
        Close();
    }

    private void OnCancel(object? sender, RoutedEventArgs e)
    {
        Result = null;
        Close();
    }
}
