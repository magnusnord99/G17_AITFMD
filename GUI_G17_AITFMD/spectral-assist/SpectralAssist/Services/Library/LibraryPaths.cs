using System.IO;

namespace SpectralAssist.Services.Library;

/// <summary>
/// Centralized path helpers for the <c>.spectral-assist/</c> sidecar that lives at the library root.
/// Runs are grouped into a subfolder per <c>imageId</c> so the sidecar stays browsable by hand.
/// Thumbnails stay flat, with one file per image.
/// <code>
/// &lt;libraryRoot&gt;/
/// └── .spectral-assist/
///     ├── library.json
///     ├── runs/
///     │   └── &lt;imageId&gt;/
///     │       └── &lt;runId&gt;.json
///     └── thumbnails/
///         └── &lt;imageId&gt;.png
/// </code>
/// </summary>
public static class LibraryPaths
{
    public const string SidecarFolder = ".spectral-assist";
    public const string ManifestFile = "library.json";
    public const string RunsFolder = "runs";
    public const string ThumbnailsFolder = "thumbnails";

    public static string Sidecar(string libraryRoot) =>
        Path.Combine(libraryRoot, SidecarFolder);

    public static string ManifestPath(string libraryRoot) =>
        Path.Combine(Sidecar(libraryRoot), ManifestFile);

    public static string Runs(string libraryRoot) =>
        Path.Combine(Sidecar(libraryRoot), RunsFolder);

    public static string Thumbnails(string libraryRoot) =>
        Path.Combine(Sidecar(libraryRoot), ThumbnailsFolder);
    
    public static string RunFolder(string libraryRoot, string imageId) =>
        Path.Combine(Runs(libraryRoot), imageId);
    
    public static string RunPath(string libraryRoot, string imageId, string runId) =>
        Path.Combine(RunFolder(libraryRoot, imageId), runId + ".json");

    public static string ThumbnailPath(string libraryRoot, string imageId) =>
        Path.Combine(Thumbnails(libraryRoot), imageId + ".png");

    /// <summary>
    /// Creates the sidecar skeleton (<c>.spectral-assist/</c>, <c>runs/</c>, <c>thumbnails/</c>).
    /// Skips and does nothing for the folders that already exist.
    /// </summary>
    public static void EnsureSidecarExists(string libraryRoot)
    {
        Directory.CreateDirectory(Sidecar(libraryRoot));
        Directory.CreateDirectory(Runs(libraryRoot));
        Directory.CreateDirectory(Thumbnails(libraryRoot));
    }
}