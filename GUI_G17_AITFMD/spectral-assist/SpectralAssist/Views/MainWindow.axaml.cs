using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using SpectralAssist.ViewModels;

namespace SpectralAssist.Views;

public partial class MainWindow : Window
{
    private MainViewModel Vm => (MainViewModel)DataContext!;

    public MainWindow()
    {
        InitializeComponent();

        DragDrop.SetAllowDrop(this, true);
        AddHandler(DragDrop.DragEnterEvent, Window_DragEnter);
        AddHandler(DragDrop.DragLeaveEvent, Window_DragLeave);
        AddHandler(DragDrop.DropEvent, Window_Drop);
    }

    private void Window_DragEnter(object? sender, DragEventArgs args)
    {
        var file = args.DataTransfer.TryGetFiles()?.FirstOrDefault();
        var valid = file != null &&
                    Path.GetExtension(file.Path.LocalPath.ToLowerInvariant()) == ".hdr";

        Vm.SetDragState(dragging: true, valid: valid);
    }

    private void Window_DragLeave(object? sender, DragEventArgs args)
    {
        Vm.SetDragState(dragging: false, valid: false);
    }

    private void Window_Drop(object? sender, DragEventArgs args)
    {
        Vm.SetDragState(dragging: false, valid: false);

        if (!args.DataTransfer.Contains(DataFormat.File)) return;

        var file = args.DataTransfer.TryGetFiles()?.FirstOrDefault();
        if (file == null) return;

        var filePath = file.Path.LocalPath;
        if (Path.GetExtension(filePath.ToLowerInvariant()) != ".hdr") return;

        // On macOS, resolve ._ resource-fork files to the actual ENVI file
        if (Path.GetFileName(filePath).StartsWith("._"))
        {
            var dir = Path.GetDirectoryName(filePath);
            var actualName = Path.GetFileName(filePath).Substring(2);
            var actualPath = !string.IsNullOrEmpty(dir) ? Path.Combine(dir, actualName) : actualName;
            if (File.Exists(actualPath))
                filePath = actualPath;
        }

        Vm.NavigateToImage(filePath);
    }


    public async void OpenHSIButton_Clicked(object sender, RoutedEventArgs args)
    {
        try
        {
            var topLevel = GetTopLevel(this);
            if (topLevel == null) return;

            var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
            {
                Title = "Open Hyperspectral Image (.hdr)",
                AllowMultiple = false,
                FileTypeFilter = [new FilePickerFileType("ENVI Header Files") { Patterns = ["*.hdr"] }]
            });

            if (files.Count < 1) return;
            var path = files[0].Path.LocalPath;
            // On macOS, skip ._ resource-fork files – use the actual ENVI file instead
            if (Path.GetFileName(path).StartsWith("._"))
            {
                var dir = Path.GetDirectoryName(path);
                var actualName = Path.GetFileName(path).Substring(2);
                var actualPath = !string.IsNullOrEmpty(dir) ? Path.Combine(dir, actualName) : actualName;
                if (File.Exists(actualPath))
                    path = actualPath;
            }

            Debug.WriteLine($"File received via file picker: {path}");
            Vm.NavigateToImage(path);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"File Picker Exception: {e.Message}");
        }
    }
}