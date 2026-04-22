using System;
using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using LibraryViewModel = SpectralAssist.ViewModels.LibraryViewModel;

namespace SpectralAssist.Views;

public partial class LibraryView : UserControl
{
    public LibraryView()
    {
        InitializeComponent();
    }

    private async void OpenFolderButton_Click(object? sender, RoutedEventArgs args)
    {
        try
        {
            if (DataContext is not LibraryViewModel vm) return;

            var top = TopLevel.GetTopLevel(this);
            if (top == null) return;

            var folders = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "Open Library Folder",
                AllowMultiple = false,
            });

            if (folders.Count > 0 && folders[0].TryGetLocalPath() is { } path)
                await vm.OpenFolderAsync(path);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"Folder Picker Exception: {e.Message}");
        }
    }
}