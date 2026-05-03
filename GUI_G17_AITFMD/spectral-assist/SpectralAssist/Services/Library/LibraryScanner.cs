using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Library;

/// <summary>
/// Recursively scans a library root and builds a folder tree. Folders that
/// contain scene .hdr files become leaf <see cref="FolderNode"/>s with
/// associated <see cref="ImageNode"/>s. Branches with no images anywhere in
/// their subtree are removed.
/// </summary>
/// <remarks>
/// The scanner currently identifies images by their paths, so moving or
/// renaming folders breaks continuity. A hash‑based approach would make the
/// process more resilient.
/// </remarks>
public class LibraryScanner
{
    private static readonly string[] DataExtensions = ["", ".raw", ".dat", ".img", ".bin"];
    private const string UncategorizedFolderName = "Uncategorized";

    /// <summary>
    /// Scans the library root and constructs a fresh <see cref="LibraryManifest"/>.
    /// Existing manifest data (if provided) is used only to preserve image IDs and
    /// notes for files whose relative paths still match. The scan walks the folder
    /// tree, identifies image folders, prunes empty branches, and groups any loose
    /// image folders into an "Uncategorized" node. No file I/O beyond directory
    /// enumeration is performed.
    /// </summary>
    /// <param name="libraryRoot">Root folder of the image library.</param>
    /// <param name="existing">Previous manifest used to retain image metadata.</param>
    /// <param name="ct">Cancellation token for aborting the scan.</param>
    /// <returns>A newly built <see cref="LibraryManifest"/>.</returns>
    /// <exception cref="DirectoryNotFoundException">Thrown if <paramref name="libraryRoot"/> does not exist.</exception>
    public static async Task<LibraryManifest> ScanAsync(string libraryRoot, LibraryManifest? existing,
        CancellationToken ct = default)
    {
        if (!Directory.Exists(libraryRoot))
            throw new DirectoryNotFoundException($"The library root {libraryRoot} was not found.");
        
        LibraryPaths.EnsureSidecarExists(libraryRoot);

        var rootFullPath = Path.GetFullPath(libraryRoot);
        var sidecarFullPath = Path.GetFullPath(LibraryPaths.Sidecar(libraryRoot)).TrimEnd(Path.DirectorySeparatorChar);
        var existingByPath = BuildPathIndex(existing);

        var topLevel = await Task.Run(() =>
        { 
            // Scan the entire directory tree under the root and collect all discovered folders
            var discovered = new List<FolderNode>();
            ScanContainer(rootFullPath, libraryRoot, relPath: "", sidecarFullPath, existingByPath, discovered, ct);
            
            // Split top-level results into container folders (no images) and loose image folders (has images)
            var containers = new List<FolderNode>(discovered.Count);
            var looseImageDirs = new List<FolderNode>();
            foreach (var node in discovered)
            {
                if (node.Images.Count > 0)
                    looseImageDirs.Add(node);
                else
                    containers.Add(node);
            }
            
            // Scan any potential loose images in the root and fill synthetic "uncategorized" collection
            var rootScenes = EnumerateSceneHdrs(rootFullPath).ToList();
            if (looseImageDirs.Count > 0 || rootScenes.Count > 0)
            {
                var uncategorized = new FolderNode
                {
                    Name = UncategorizedFolderName,
                    CurrentRelPath = string.Empty,
                    Children = looseImageDirs,
                    Images = rootScenes.Count > 0
                        ? BuildImageFolder(rootFullPath, libraryRoot, relPath: string.Empty,
                            name: UncategorizedFolderName, rootScenes, existingByPath).Images
                        : [],
                };
                containers.Add(uncategorized);
            }

            SortTree(containers);
            return containers;
        }, ct);

        return new LibraryManifest
        {
            LastScanned = DateTime.UtcNow,
            Folders = topLevel,
        };
    }

    /// <summary>
    /// Recursively scans a container directory to build its folder hierarchy.
    /// Image folders (those containing scene .hdr files) become leaf nodes, while
    /// pure container folders recurse. Empty branches are pruned.
    /// </summary>
    /// <param name="absPath">Absolute path of the container being scanned.</param>
    /// <param name="libraryRoot">Root of the library for computing relative paths.</param>
    /// <param name="relPath">Relative path of the current folder.</param>
    /// <param name="sidecarFullPath">Full path of the sidecar folder to exclude.</param>
    /// <param name="existingByPath">Lookup of existing ImageNodes to preserve IDs/notes.</param>
    /// <param name="outChildren">Collection to receive discovered child folders.</param>
    /// <param name="ct">Cancellation token for aborting the scan.</param>
    private static void ScanContainer(string absPath, string libraryRoot, string relPath, string sidecarFullPath,
        Dictionary<string, ImageNode> existingByPath, List<FolderNode> outChildren, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        IEnumerable<string> subDirs;
        try
        {
            subDirs = Directory.EnumerateDirectories(absPath);
        }
        catch
        {
            return;
        }

        foreach (var subDir in subDirs.OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
        {
            var subName = Path.GetFileName(subDir);
            if (ShouldSkipDirectory(subName, subDir, sidecarFullPath)) continue;

            var childRel = string.IsNullOrEmpty(relPath) ? subName : relPath + "/" + subName;
            var scenes = EnumerateSceneHdrs(subDir).ToList();

            if (scenes.Count > 0)
            {
                // Image folder: stop recurse.
                outChildren.Add(BuildImageFolder(subDir, libraryRoot, childRel, subName,
                    scenes, existingByPath));
            }
            else
            {
                // Container folder: recurse, then only keep if subtree has images.
                var grandChildren = new List<FolderNode>();
                ScanContainer(subDir, libraryRoot, childRel, sidecarFullPath,
                    existingByPath, grandChildren, ct);

                // Prune empty branches
                if (grandChildren.Count == 0) 
                    continue;

                outChildren.Add(new FolderNode
                {
                    Name = subName,
                    CurrentRelPath = childRel,
                    Children = grandChildren,
                });
            }
        }
    }


    /// <summary>
    /// Builds a <see cref="FolderNode"/> for the given folder, creating ImageNodes
    /// for each scene .hdr file and preserving existing metadata when available.
    /// </summary>
    /// <param name="absPath">Absolute path of the folder.</param>
    /// <param name="libraryRoot">Root of the library for computing relative paths.</param>
    /// <param name="relPath">Folder's relative path.</param>
    /// <param name="name">Folder name.</param>
    /// <param name="sceneHdrs">Scene .hdr files contained in the folder.</param>
    /// <param name="existingByPath">Lookup of existing ImageNodes to reuse IDs and notes.</param>
    /// <returns>A populated <see cref="FolderNode"/>.</returns>
    private static FolderNode BuildImageFolder(string absPath, string libraryRoot, string relPath, string name,
        List<string> sceneHdrs, Dictionary<string, ImageNode> existingByPath)
    {
        var hasCalibration = DetectCalibration(absPath);
        var images = new List<ImageNode>(sceneHdrs.Count);

        images.AddRange(from hdrPath in sceneHdrs.OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            let imgRelPath = RelativePath(libraryRoot, hdrPath)
            let prior = existingByPath.GetValueOrDefault(imgRelPath)
            select new ImageNode
            {
                ImageId = prior?.ImageId ?? NewImageId(),
                CurrentRelPath = imgRelPath,
                SceneFileName = Path.GetFileName(hdrPath),
                HasCalibration = hasCalibration,
                Notes = prior?.Notes ?? string.Empty,
                Runs = prior?.Runs ?? []
            });

        return new FolderNode
        {
            Name = name,
            CurrentRelPath = relPath,
            Images = images,
        };
    }


    /// <summary>
    /// Enumerates all valid scene .hdr files in the specified folder. The search
    /// is non‑recursive and filters out hidden/system files, calibration files
    /// (dark/white), and any .hdr that does not have a corresponding data file.
    /// </summary>
    /// <param name="folder">The folder to scan for scene .hdr files.</param>
    /// <returns>Full paths to .hdr files that qualify as scene files.</returns>
    private static IEnumerable<string> EnumerateSceneHdrs(string folder)
    {
        string[] hdrFiles;
        try
        {
            hdrFiles = Directory.GetFiles(folder, "*.hdr", SearchOption.TopDirectoryOnly);
        }
        catch
        {
            yield break;
        }

        foreach (var file in hdrFiles)
        {
            var fileName = Path.GetFileName(file);
            if (fileName.StartsWith("._")) continue;

            var baseName = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
            if (baseName.Contains("dark") || baseName.Contains("white")) continue;

            if (TryFindDataFile(file) == null) continue;

            yield return Path.GetFullPath(file);
        }
    }


    /// <summary>
    /// Determines whether a directory should be skipped during traversal.
    /// Skips hidden folders (names starting with '.') and the folder that
    /// matches the resolved sidecar directory path.
    /// </summary>
    /// <param name="name">The directory name (not the full path).</param>
    /// <param name="fullPath">The full path to the directory being evaluated.</param>
    /// <param name="sidecarFullPath">The full path of the sidecar directory to exclude.</param>
    /// <returns>True if the directory should be ignored; otherwise false.</returns>
    private static bool ShouldSkipDirectory(string name, string fullPath, string sidecarFullPath)
    {
        if (name.StartsWith('.')) return true;
        var full = Path.GetFullPath(fullPath).TrimEnd(Path.DirectorySeparatorChar);
        return string.Equals(full, sidecarFullPath, StringComparison.OrdinalIgnoreCase);
    }


    /// <summary>
    /// Attempts to locate a data file that shares the same base name as the given .hdr file by checking
    /// all known data file extensions. Returns the first matching file that exists on disk.
    /// </summary>
    /// <param name="hdrPath"> The path to the .hdr file whose companion data file should be searched for.</param>
    /// <returns>The full path to the first matching data file, or null if none of the expected extensions are found.</returns>
    private static string? TryFindDataFile(string hdrPath)
    {
        var dirPath = Path.GetDirectoryName(hdrPath) ?? string.Empty;
        var fileName = Path.GetFileNameWithoutExtension(hdrPath);

        return DataExtensions.Select(fileExt => Path.Combine(dirPath, fileName + fileExt)).FirstOrDefault(File.Exists);
    }


    /// <summary>
    /// Checks the directory of the given scene header file for calibration assets by
    /// checking the existence of a "dark" and "white" .hdr file in the same folder.
    /// </summary>
    /// <param name="folder">Folder of the header file</param>
    /// <returns>Returns true only if both a "dark" and a "white" .hdr file is present.</returns>
    private static bool DetectCalibration(string folder)
    {
        bool hasDark = false, hasWhite = false;
        IEnumerable<string> files;
        try
        {
            files = Directory.EnumerateFiles(folder, "*.hdr");
        }
        catch
        {
            return false;
        }

        foreach (var file in files)
        {
            var name = Path.GetFileNameWithoutExtension(file).ToLowerInvariant();
            if (name.Contains("dark"))
                hasDark = true;
            else if (name.Contains("white"))
                hasWhite = true;

            if (hasDark && hasWhite) return true;
        }

        return false;
    }

    /// <summary>
    /// Recursively sorts the folder tree in place. Folders at each level are
    /// ordered alphabetically using a case‑insensitive comparison, and the same
    /// ordering is applied to all descendant folders.
    /// </summary>
    /// <param name="folders">The list of FolderNode objects to sort, including all nested children.</param>
    private static void SortTree(List<FolderNode> folders)
    {
        folders.Sort((a, b) => StringComparer.OrdinalIgnoreCase.Compare(a.Name, b.Name));
        foreach (var folder in folders) SortTree(folder.Children);
    }

    /// <summary>
    /// Creates a case‑insensitive index mapping each image's relative path to its
    /// corresponding ImageNode. Used for fast lookup of images by path.
    /// </summary>
    /// <param name="manifest">
    /// The library manifest whose folder tree will be scanned for images.
    /// If null, an empty dictionary is returned.
    /// </param>
    /// <returns>A dictionary keyed by relative image paths, containing all ImageNodes found in the manifest.</returns>
    private static Dictionary<string, ImageNode> BuildPathIndex(LibraryManifest? manifest)
    {
        var dict = new Dictionary<string, ImageNode>(StringComparer.OrdinalIgnoreCase);
        if (manifest == null) return dict;

        foreach (var image in FlattenImages(manifest.Folders))
            if (!string.IsNullOrEmpty(image.CurrentRelPath))
                dict[image.CurrentRelPath] = image;

        return dict;
    }

    /// <summary>
    /// Recursively enumerates all ImageNodes contained within the given folder
    /// hierarchy, yielding images from each folder and its children.
    /// </summary>
    /// <param name="folders">Root folders to search for images</param>
    /// <returns></returns>
    public static IEnumerable<ImageNode> FlattenImages(IEnumerable<FolderNode> folders)
    {
        foreach (var folder in folders)
        {
            foreach (var image in folder.Images) yield return image;
            foreach (var image in FlattenImages(folder.Children)) yield return image;
        }
    }


    private static string RelativePath(string root, string fullPath) =>
        Path.GetRelativePath(root, fullPath).Replace('\\', '/');

    private static string NewImageId() =>
        "img_" + Guid.NewGuid().ToString("N")[..16];
}