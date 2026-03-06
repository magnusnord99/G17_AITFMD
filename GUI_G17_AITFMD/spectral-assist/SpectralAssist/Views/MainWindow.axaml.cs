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
        if (Path.GetExtension(filePath.ToLowerInvariant()) == ".hdr")
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
            Debug.WriteLine($"File received via file picker: {files[0].Path.LocalPath}");
            Vm.NavigateToImage(files[0].Path.LocalPath);
        }
        catch (Exception e)
        {
            Debug.WriteLine($"File Picker Exception: {e.Message}");
        }
    }
    
    
    
    
    
    
    /*
     * 

    
    public void Window_DragOver(object sender, DragEventArgs args)
    {
        var file = args.DataTransfer.TryGetFiles()?.FirstOrDefault();
        var extension = Path.GetExtension(file?.Path.LocalPath.ToLowerInvariant());

        if (file != null && extension == ".hdr")
        {
            
        }

        args.DragEffects = valid ? DragDropEffects.Copy : DragDropEffects.None;
        Vm.DropZoneText = valid ? "Drop file to load HSI" : "File not supported: please use a .hdr file";
    }

    public void DropZone_DragLeave(object sender, DragEventArgs args)
    {
        Vm.DropZoneText = "Drag and drop .hdr file here";
    }
    
    public void DropZone_Drop(object sender, DragEventArgs args)
    {
        if (!args.DataTransfer.Contains(DataFormat.File)) return;
        
        var file = args.DataTransfer.TryGetFiles()?.FirstOrDefault();
        if (file == null) return;
        
        var filePath = file.Path.LocalPath;
        if (Path.GetExtension(filePath.ToLowerInvariant()) == ".hdr")
            Vm.NavigateToImage(filePath);
        Vm.LoadHsiFile(filePath);
        
        // Check file extension
        var filePath = file.Path.LocalPath;
        var fileExtension = Path.GetExtension(filePath.ToLowerInvariant());
        if (fileExtension == ".hdr")
        {
            Debug.WriteLine($"File received via drag-and-drop: {filePath}");
            // LoadHsiFile(filePath)
        }
    }
         */
    
}