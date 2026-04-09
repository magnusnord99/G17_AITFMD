using System;
using System.Diagnostics;
using System.IO;
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

            var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Select Model Package (manifest.json)",
                AllowMultiple = false,
                FileTypeFilter = [new FilePickerFileType("Model Manifest") { Patterns = ["manifest.json"] }]
            });

            if (files.Count < 1) return;

            var packageDir = Path.GetDirectoryName(files[0].Path.LocalPath);
            if (packageDir == null) return;

            Debug.WriteLine($"Import model package: {packageDir}");

            if (DataContext is ModelsViewModel vm)
                vm.PreviewImport(packageDir);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"Import Model Exception: {e.Message}");
        }
    }
}