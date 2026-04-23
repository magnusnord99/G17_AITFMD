using System;
using System.IO;
using System.Linq;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Extensions;
using SpectralAssist.Models;
using SpectralAssist.Services.Library;

namespace SpectralAssist.ViewModels.Components;

public partial class ImageTileViewModel : ObservableObject
{
    public ImageTileViewModel(ImageNode source, string libraryRoot, Action<ImageNode> onOpen)
    {
        Source = source;
        _onOpen = onOpen;
        Thumbnail = ThumbnailService.TryLoadThumbnail(libraryRoot, source);
    }
    
    [RelayCommand]
    private void Open() => _onOpen(Source);

    // States __________________________________________________________
    private readonly Action<ImageNode> _onOpen;
    [ObservableProperty] private ImageNode _source;
    [ObservableProperty] private Bitmap? _thumbnail;
    [ObservableProperty] private bool _isActive;
    
    // Properties ______________________________________________________
    public string FileName => Source.SceneFileName;
    public bool MissingCalibration => !Source.HasCalibration;
    public int ReportCount => Source.Runs.Count;
    public bool HasReports => ReportCount > 0;
    public bool IsNotAnalyzed => ReportCount == 0;
    
    public void RefreshActive(string? activeImageId)
        => IsActive = activeImageId is not null && Source.ImageId == activeImageId;
    
    public string DisplayName
    {
        get
        {
            var parent = Path.GetDirectoryName(Source.CurrentRelPath)?.TrimEnd('/', '\\');
            var folder = string.IsNullOrEmpty(parent) ? null : Path.GetFileName(parent);

            return !string.IsNullOrEmpty(folder)
                ? folder
                : Path.GetFileNameWithoutExtension(Source.SceneFileName);
        }
    }

    public string PrimarySummary
    {
        get
        {
            var mostSevereRun = Source.MostSevereRun();
            if (mostSevereRun == null)
                return "Not analyzed yet";

            var name = string.IsNullOrWhiteSpace(mostSevereRun.PositiveClassName)
                ? "positive"
                : mostSevereRun.PositiveClassName;

            return $"Peak: {mostSevereRun.PositiveClassPercentAbove50:0}% {name}";
        }
    }

    public string SecondarySummary
    {
        get
        {
            var latestRun = Source.LatestRun();
            var date = latestRun?.DatePerformed.ToLocalTime().ToString("MMM d");
            return ReportCount == 1
                ? $"1 report · {date}"
                : $"{ReportCount} reports · Last {date}";
        }
    }
}