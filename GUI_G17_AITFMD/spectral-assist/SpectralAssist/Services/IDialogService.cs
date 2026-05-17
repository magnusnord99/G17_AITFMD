using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services;

/// <summary>
/// Application-level dialog gateway. Lets view-models request user input
/// without referencing concrete <c>Window</c> types from the View layer.
/// </summary>
public interface IDialogService
{
    /// <summary>
    /// Show a yes/no confirmation dialog. Returns true if the user confirmed.
    /// </summary>
    Task<bool> ConfirmAsync(
        string title,
        string message,
        string confirmLabel = "Delete",
        string cancelLabel = "Cancel",
        bool isDestructive = true);

    /// <summary>
    /// Show the PDF export options dialog. Returns the chosen
    /// <see cref="ExportOptions"/>, or null if the user canceled.
    /// </summary>
    Task<ExportOptions?> ShowExportDialogAsync(ExportOptions? initial = null);
}
