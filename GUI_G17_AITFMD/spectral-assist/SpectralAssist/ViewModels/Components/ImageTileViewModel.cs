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
    
    //__ States __________________________________________________________
    private readonly Action<ImageNode> _onOpen;
    [ObservableProperty] private ImageNode _source;
    [ObservableProperty] private Bitmap? _thumbnail;
    [ObservableProperty] private bool _isActive;
    
    //__ Properties ______________________________________________________
    private RunSummary? MostSevereRun => Source.MostSevereRun();
    private RunSummary? LatestRun => Source.LatestRun();
    public string FileName => Source.SceneFileName;
    public bool MissingCalibration => !Source.HasCalibration;
    
    public int ReportCount => Source.Runs.Count;
    public bool HasReports => ReportCount > 0;
    public bool IsNotAnalyzed => ReportCount == 0;
    
    public void RefreshActive(string? activeImageId)
        => IsActive = activeImageId is not null && Source.ImageId == activeImageId;
    
    public double PositiveClassPercentAbove50 =>
        MostSevereRun?.PositiveClassPercentAbove50 ?? 0;

    public double PositiveClassPercentAbove80 =>
        MostSevereRun?.PositiveClassPercentAbove80 ?? 0;

    public string PositiveClassName =>
        string.IsNullOrWhiteSpace(MostSevereRun?.PositiveClassName)
            ? "positive"
            : MostSevereRun!.PositiveClassName;
    
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

    public string FooterSummary
    {
        get
        {
            if (LatestRun is null)
                return $"{ReportCount} reports";
            
            var date = LatestRun.CompletedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm");

            return ReportCount == 1
                ? $"1 report - Last run {date}"
                : $"{ReportCount} reports - Last run {date}";
        }
    }
    
    //__ Methods ____________________________________________
    
    [RelayCommand]
    private void Open() => _onOpen(Source);
    
    public void UpdateFrom(ImageNode updated, string libraryRoot)
    {
        Source = updated;
        Thumbnail = ThumbnailService.TryLoadThumbnail(libraryRoot, updated);

        OnPropertyChanged(nameof(FileName));
        OnPropertyChanged(nameof(MissingCalibration));
        OnPropertyChanged(nameof(ReportCount));
        OnPropertyChanged(nameof(HasReports));
        OnPropertyChanged(nameof(IsNotAnalyzed));
        OnPropertyChanged(nameof(DisplayName));
        OnPropertyChanged(nameof(FooterSummary));
    }
}