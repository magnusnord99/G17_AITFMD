using System;
using System.Collections.Generic;

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
    public List<RunSummary> Runs { get; set; } = [];
}

public class RunSummary
{
    public string RunId { get; init; } = string.Empty;
    public string ModelName { get; init; } = string.Empty;
    public DateTime DatePerformed { get; init; }
    public string PositiveClassName { get; init; } = string.Empty;
    public double PositiveClassPercentAbove50 { get; init; }
    public double PositiveClassPercentAbove80 { get; init; }
}