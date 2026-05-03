using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json.Serialization;

namespace SpectralAssist.Models;

public class LibraryManifest
{
    public DateTime LastScanned { get; set; }
    public List<FolderNode> Folders { get; set; } = [];
}

public class FolderNode
{
    public string Name { get; set; } = string.Empty;
    public string CurrentRelPath { get; set; } = string.Empty;
    public List<FolderNode> Children { get; set; } = [];
    public List<ImageNode> Images { get; set; } = [];
}

public class ImageNode
{
    public string ImageId { get; set; } = string.Empty;
    public string CurrentRelPath { get; set; } = string.Empty;
    public string SceneFileName { get; set; } = string.Empty;
    public bool HasCalibration { get; set; }
    public string Notes { get; set; } = string.Empty;
    public ObservableCollection<RunSummary> Runs { get; set; } = [];

    [JsonIgnore] public string AbsolutePath { get; set; } = string.Empty;
    [JsonIgnore] public bool IsInLibrary => !string.IsNullOrEmpty(ImageId);

    /// <summary>
    /// Creates a temporary (transient) ImageNode for single-file mode.
    /// Used for single image node (meaning no library or persistence).
    /// </summary>
    /// <param name="absPath">the absolute path to the image's header file (.hdr)</param>
    /// <returns>A transient <c>ImageNode</c> with empty ImageId</returns>
    public static ImageNode CreateTransient(string absPath) => new()
    {
        ImageId = string.Empty,
        CurrentRelPath = Path.Combine(Path.GetFileName(Path.GetDirectoryName(absPath)) ?? string.Empty,
            Path.GetFileName(absPath)),
        SceneFileName = Path.GetFileName(absPath),
        HasCalibration = false,
        AbsolutePath = absPath,
    };
}

public class RunSummary
{
    public string RunId { get; init; } = string.Empty;
    public string ModelDisplayName { get; init; } = string.Empty;
    public DateTime CompletedAt { get; init; }
    public string PositiveClassName { get; init; } = string.Empty;
    public double PositiveClassPercentAbove50 { get; init; }
    public double PositiveClassPercentAbove80 { get; init; }
}