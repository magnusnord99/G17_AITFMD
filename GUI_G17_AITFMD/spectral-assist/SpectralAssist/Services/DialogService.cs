using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using SpectralAssist.Models;
using SpectralAssist.ViewModels;
using SpectralAssist.Views;
using ConfirmDialog = SpectralAssist.Views.Components.ConfirmDialog;

namespace SpectralAssist.Services;

/// <summary>
/// Default <see cref="IDialogService"/> implementation. Dialogs follow the same
/// pattern across the app: a custom Window paired with a ViewModel that receives
/// the Window so it can call <c>dialog.Close(result)</c>.
/// </summary>
public class DialogService : IDialogService
{
    private static Window? OwnerWindow =>
        (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

    public async Task<bool> ConfirmAsync(
        string title,
        string message,
        string confirmLabel = "Delete",
        string cancelLabel = "Cancel",
        bool isDestructive = true)
    {
        var owner = OwnerWindow;
        if (owner is null) return false;

        var dialog = new ConfirmDialog();
        dialog.DataContext = new ConfirmDialogViewModel(
            dialog, title, message, cancelLabel, confirmLabel, isDestructive);

        var result = await dialog.ShowDialog<bool?>(owner);
        return result == true;
    }

    public async Task<ExportOptions?> ShowExportDialogAsync(ExportOptions? initial = null)
    {
        var owner = OwnerWindow;
        if (owner is null) return null;

        var dialog = new ExportDialog();
        dialog.DataContext = new ExportDialogViewModel(dialog, initial);

        return await dialog.ShowDialog<ExportOptions?>(owner);
    }
}
