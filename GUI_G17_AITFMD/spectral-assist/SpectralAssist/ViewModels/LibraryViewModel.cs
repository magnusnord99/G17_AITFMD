using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Extensions;
using SpectralAssist.Models;
using SpectralAssist.Services.Library;
using SpectralAssist.ViewModels.Components;

namespace SpectralAssist.ViewModels;

public enum LibraryState { Empty, Scanning, Loaded, Failed }

public partial class LibraryViewModel : ViewModelBase
{
    private const string UncategorizedName = "Uncategorized";

    private readonly LibraryManager _manager;
    private readonly Action<ImageNode> _openImage;
    private readonly Dictionary<string, Components.ImageTileViewModel> _tileCache = new();

    private CancellationTokenSource? _scanCts;
    private List<ImageNode> _allImagesCache = [];

    public LibraryViewModel(LibraryManager manager, Action<ImageNode> openImage)
    {
        _manager = manager;
        _openImage = openImage;
        if (_manager.IsOpen) PopulateFromManifest();
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

    
    // Tree and Gallery Stuff ___________________________________
    public ObservableCollection<LibraryTreeItem> TreeRoots { get; } = [];
    public ObservableCollection<ImageTileViewModel> CurrentImages { get; } = [];

    [ObservableProperty] private LibraryTreeItem? _selectedTreeItem;
    [ObservableProperty] private string? _activeImageId;

    partial void OnActiveImageIdChanged(string? value)
    {
        foreach (var tile in CurrentImages)
            tile.RefreshActive(value);
    }

    partial void OnSelectedTreeItemChanged(LibraryTreeItem? value)
    {
        CurrentImages.Clear();
        if (value == null || _manager.Root == null) return;

        // A branch shows every image in its subtree; a leaf image-folder shows just its own.
        foreach (var image in LibraryScanner.FlattenImages([value.Source]))
            CurrentImages.Add(GetOrCreateTile(image));
    }

    private ImageTileViewModel GetOrCreateTile(ImageNode image)
    {
        if (!_tileCache.TryGetValue(image.ImageId, out var tile))
        {
            tile = new ImageTileViewModel(image, _manager.Root!, _openImage);
            _tileCache[image.ImageId] = tile;
        }

        tile.RefreshActive(ActiveImageId);
        return tile;
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
            StatusMessage = $"Loaded: {PatientCount} folder(s), {ScanCount} image(s)";
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
        _tileCache.Clear();
        _allImagesCache = [];

        SelectedTreeItem = null;
        StatusMessage = string.Empty;
        State = LibraryState.Empty;
        OnPropertyChanged(nameof(LibraryRoot));
        NotifySummaryChanged();
    }
    
    
    public void RefreshView()
    {
        if (_manager.IsOpen) PopulateFromManifest();
    }

    
    
    // Population from Manifest _______________________________________________
    private void PopulateFromManifest()
    {
        var previousPath = SelectedTreeItem?.RelPath;

        TreeRoots.Clear();
        CurrentImages.Clear();

        var folders = _manager.Manifest?.Folders ?? (IReadOnlyList<FolderNode>)Array.Empty<FolderNode>();
        _allImagesCache = LibraryScanner.FlattenImages(folders).ToList();

        // Drop cached tiles for images that no longer exist (e.g. after rescan).
        var liveIds = _allImagesCache.Select(i => i.ImageId).ToHashSet();
        foreach (var stale in _tileCache.Keys.Where(id => !liveIds.Contains(id)).ToList())
            _tileCache.Remove(stale);

        foreach (var f in folders)
            TreeRoots.Add(new LibraryTreeItem(f));

        SelectedTreeItem = previousPath != null
            ? FindByRelPath(TreeRoots, previousPath) ?? TreeRoots.FirstOrDefault()
            : TreeRoots.FirstOrDefault();

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

    private static LibraryTreeItem? FindByRelPath(IEnumerable<LibraryTreeItem> items, string relPath)
    {
        foreach (var item in items)
        {
            if (item.RelPath == relPath) return item;
            var hit = FindByRelPath(item.Children, relPath);
            if (hit != null) return hit;
        }

        return null;
    }
}

/// <summary>
/// Viewmodel wrapper around a <see cref="FolderNode"/> used to populate the
/// library tree. Each folder becomes a tree item: folders with children appear
/// as expandable nodes, while leaf folders (those containing only images)
/// appear as non‑expandable items.
/// </summary>
public sealed class LibraryTreeItem
{
    public FolderNode Source { get; }
    public string Name => Source.Name;
    public string RelPath => Source.CurrentRelPath;
    public int ImageCount  => Source.TotalImageCount();
    public int ReportCount => Source.TotalReportCount();
    public ObservableCollection<LibraryTreeItem> Children { get; }
    public LibraryTreeItem(FolderNode source)
    {
        Source = source;
        Children = new ObservableCollection<LibraryTreeItem>(
            source.Children
                .Where(c => c.Children.Count > 0)
                .Select(c => new LibraryTreeItem(c)));
    }
}