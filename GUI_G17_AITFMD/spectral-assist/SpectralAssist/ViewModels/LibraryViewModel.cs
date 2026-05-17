using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Extensions;
using SpectralAssist.Models;
using SpectralAssist.Services;
using SpectralAssist.Services.Library;
using SpectralAssist.ViewModels.Components;

namespace SpectralAssist.ViewModels;

public enum LibraryState
{
    Empty,
    Scanning,
    Loaded,
    Failed
}

public partial class LibraryViewModel : ViewModelBase
{
    private const string UncategorizedName = "Uncategorized";

    private readonly LibraryManager _manager;

    private SessionService Session { get; }
    private List<ImageNode> _allImagesCache = [];
    private CancellationTokenSource? _scanCts;

    public event Action<ImageNode>? ImageSelected;

    public LibraryViewModel(LibraryManager manager, SessionService session)
    {
        _manager = manager;
        Session = session;
        _manager.ImageUpdated += OnImageUpdated;
        session.PropertyChanged += OnSessionChanged;
    }

    // Library State _______________________________________________________________
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsEmpty), nameof(IsScanning), nameof(IsLoaded), nameof(IsFailed))]
    [NotifyCanExecuteChangedFor(nameof(RescanCommand), nameof(CloseLibraryCommand))]
    private LibraryState _state = LibraryState.Empty;

    public bool IsEmpty => State == LibraryState.Empty;
    public bool IsScanning => State == LibraryState.Scanning;
    public bool IsLoaded => State == LibraryState.Loaded;
    public bool IsFailed => State == LibraryState.Failed;

    [ObservableProperty] private string _statusMessage = string.Empty;

    public string? LibraryRoot => _manager.Root;

    // Header Summary _____________________________________
    public string DatasetName =>
        string.IsNullOrEmpty(_manager.Root)
            ? string.Empty
            : Path.GetFileName(_manager.Root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

    public int PatientCount => _manager.Manifest?.Folders.Count(f => f.Name != UncategorizedName) ?? 0;
    public int ScanCount => _allImagesCache.Count;
    public int ReportedScanCount => _allImagesCache.Count(i => i.Runs.Count > 0);
    public int MissingCalibrationCount => _allImagesCache.Count(i => !i.HasCalibration);


    // Tree and Image Tiles ___________________________________
    public ObservableCollection<LibraryTreeItem> TreeRoots { get; } = [];
    public ObservableCollection<ImageTileViewModel> CurrentImages { get; } = [];
    
    private LibraryTreeItem? _selectedTreeItem;
    public LibraryTreeItem? SelectedTreeItem
    {
        get => _selectedTreeItem;
        set
        {
            if (value == null && _selectedTreeItem != null) return;
            if (!SetProperty(ref _selectedTreeItem, value)) return;
            OnSelectedTreeItemChanged(value);
        }
    }
    
    private void OnTileClicked(ImageNode node) => ImageSelected?.Invoke(node);

    private void OnSelectedTreeItemChanged(LibraryTreeItem? value)
    {
        CurrentImages.Clear();
        if (value == null) return;

        foreach (var image in LibraryScanner.FlattenImages([value.Source]))
        {
            var tile = new ImageTileViewModel(image, _manager.Root!, OnTileClicked);
            tile.RefreshActive(Session.ActiveImageId);
            CurrentImages.Add(tile);
        }
    }

    private void OnSessionChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(SessionService.ActiveImageId))
            foreach (var tile in CurrentImages)
                tile.RefreshActive(Session.ActiveImageId);
    }

    // ---- Commands ----
    public async Task OpenFolderAsync(string rootPath)
    {
        // Stop existing scan if already in progress.
        _scanCts?.Cancel();
        _scanCts = new CancellationTokenSource();

        State = LibraryState.Scanning;
        StatusMessage = $"Scanning {rootPath}…";
        OnPropertyChanged(nameof(LibraryRoot));

        try
        {
            await _manager.OpenAsync(rootPath, _scanCts.Token);
            PopulateFromManifest();
            State = LibraryState.Loaded;
            StatusMessage = $"Loaded: {PatientCount} folders(s), {ScanCount} image(s)";
        }
        catch (OperationCanceledException)
        {
            State = LibraryState.Empty;
            StatusMessage = "Scan cancelled";
        }
        catch (Exception ex)
        {
            State = LibraryState.Failed;
            StatusMessage = $"Scan failed: {ex.Message}";
        }
        finally
        {
            OnPropertyChanged(nameof(LibraryRoot));
        }
    }

    [RelayCommand(CanExecute = nameof(IsLoaded))]
    private async Task RescanAsync()
    {
        if (_manager.Root == null) return;

        State = LibraryState.Scanning;
        StatusMessage = "Rescanning…";
        try
        {
            await _manager.RescanAsync();
            PopulateFromManifest();
            State = LibraryState.Loaded;
            StatusMessage = $"Rescan complete: {ScanCount} image(s)";
        }
        catch (Exception ex)
        {
            State = LibraryState.Failed;
            StatusMessage = $"Rescan failed: {ex.Message}";
        }
    }

    [RelayCommand(CanExecute = nameof(IsLoaded))]
    private void CloseLibrary()
    {
        _scanCts?.Cancel();
        _manager.Close();

        TreeRoots.Clear();
        CurrentImages.Clear();
        _allImagesCache.Clear();
        
        if (SetProperty(ref _selectedTreeItem, null, nameof(SelectedTreeItem)))
            OnSelectedTreeItemChanged(null);
        
        StatusMessage = string.Empty;
        State = LibraryState.Empty;

        OnPropertyChanged(nameof(LibraryRoot));
        NotifySummaryChanged();
    }
    
    
    private void OnImageUpdated(ImageNode updated)
    {
        for (var i = 0; i < _allImagesCache.Count; i++)
        {
            if (_allImagesCache[i].ImageId == updated.ImageId)
            {
                _allImagesCache[i] = updated;
                break;
            }
        }

        foreach (var tile in CurrentImages.Where(t => t.Source.ImageId == updated.ImageId))
            tile.UpdateFrom(updated, _manager.Root!);

        NotifySummaryChanged();
    }
    

    /// <summary>
    /// Rebuilds <see cref="TreeRoots"/> and <see cref="CurrentImages"/> from the in-memory manifest,
    /// preserving the currently-selected node across rescans/refreshes.
    /// </summary>
    private void PopulateFromManifest()
    {
        TreeRoots.Clear();
        CurrentImages.Clear();
        _allImagesCache = _manager.Manifest != null
            ? LibraryScanner.FlattenImages(_manager.Manifest.Folders).ToList()
            : [];

        if (_manager.Manifest != null) 
            foreach (var f in _manager.Manifest.Folders) 
                TreeRoots.Add(new LibraryTreeItem(f));
        
        SelectedTreeItem = TreeRoots.FirstOrDefault();
        NotifySummaryChanged();
    }
    
    private void NotifySummaryChanged()
    {
        OnPropertyChanged(nameof(DatasetName));
        OnPropertyChanged(nameof(PatientCount));
        OnPropertyChanged(nameof(ScanCount));
        OnPropertyChanged(nameof(ReportedScanCount));
        OnPropertyChanged(nameof(MissingCalibrationCount));
    }
}

/// <summary>
/// Viewmodel wrapper around a <see cref="FolderNode"/> used to populate the
/// library tree. Each folder becomes a tree item: folders with children appear
/// as expandable nodes, while leaf folders (those containing only images)
/// appear as non‑expandable items.
/// </summary>
public sealed partial class LibraryTreeItem : ObservableObject
{
    public FolderNode Source { get; }
    public string Name => Source.Name;
    public string RelPath => Source.CurrentRelPath;
    public int ImageCount => Source.TotalImageCount();
    public int ReportCount => Source.TotalReportCount();
    public ObservableCollection<LibraryTreeItem> Children { get; }

    [ObservableProperty] private bool _isExpanded;

    public LibraryTreeItem(FolderNode source)
    {
        Source = source;
        Children = new ObservableCollection<LibraryTreeItem>(
            source.Children
                .Where(c => c.Children.Count > 0)
                .Select(c => new LibraryTreeItem(c)));
    }
}