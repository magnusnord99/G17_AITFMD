using Avalonia.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace SpectralAssist.ViewModels;

public partial class ConfirmDialogViewModel(
    Window dialog,
    string dialogTitle,
    string message,
    string cancelButtonText,
    string confirmButtonText,
    bool isDestructive = false) : ObservableObject
{
    public string DialogTitle { get; } = dialogTitle;
    public string Message { get; } = message;
    public string CancelButtonText { get; } = cancelButtonText;
    public string ConfirmButtonText { get; } = confirmButtonText;

    /// <summary>
    /// When true, the Confirm button gets the <c>danger</c> style (red);
    /// otherwise it gets the <c>primary</c> style (accent).
    /// XAML binds <c>Classes.danger</c> and <c>Classes.primary</c> to this flag.
    /// </summary>
    public bool IsDestructive { get; } = isDestructive;

    [RelayCommand]
    private void Confirm() => dialog.Close(true);

    [RelayCommand]
    private void Cancel() => dialog.Close(false);
}
