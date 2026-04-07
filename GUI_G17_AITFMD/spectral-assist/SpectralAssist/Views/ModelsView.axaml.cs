using System;
using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class ModelsView : UserControl
{
    public ModelsView()
    {
        InitializeComponent();
    }
    
    public async void ImportModelButton_Clicked(object sender, RoutedEventArgs args)
    {
        try
        {
            var topLevel = TopLevel.GetTopLevel(this);
            if (topLevel == null) return;

            var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(
                new FolderPickerOpenOptions
                {
                    Title = "Select Model Package Folder (must contain manifest.json + model.onnx)",
                    AllowMultiple = false,
                });

            if (folders.Count < 1) return;

            var folderPath = folders[0].Path.LocalPath;
            Debug.WriteLine($"Import model folder selected: {folderPath}");

            if (DataContext is ModelsViewModel vm)
                vm.PreviewImport(folderPath);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"Import Model Exception: {e.Message}");
        }
    }
}