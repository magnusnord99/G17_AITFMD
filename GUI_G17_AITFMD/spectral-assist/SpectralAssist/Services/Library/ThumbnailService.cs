using System;
using System.IO;
using System.Linq;
using Avalonia;
using Avalonia.Media.Imaging;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Library;

/// <summary>
/// Provides best-effort thumbnail loading and saving for images in a library.
/// Thumbnails are stored in the library’s sidecar folder and are used to speed
/// up UI rendering. All operations are safe to call repeatedly and failures are
/// silently ignored.
/// </summary>
public static class ThumbnailService
{
    // Maximum size of the thumbnail’s longest edge, in pixels.
    private const int MaxEdge = 320;
    
    // Valid preview image file extensions
    private static readonly string[] PreviewExtensions = [".jpg", ".jpeg", ".png", ".bmp"];
    
    /// <summary>
    /// Attempts to load a thumbnail bitmap for the given image. The method resolves
    /// the thumbnail path using the service’s standard lookup rules (cached
    /// thumbnail, then sibling preview) and loads the bitmap if possible.
    /// Returns <c>null</c> if no thumbnail can be resolved or loaded.
    /// </summary>
    /// <param name="libraryRoot">The library root to load the thumbnail from.</param>
    /// <param name="image">The image node whose thumbnail is being loaded.</param>
    /// <returns>The loaded bitmap, or <c>null</c> if unavailable.</returns>
    public static Bitmap? TryLoadThumbnail(string libraryRoot, ImageNode image)
    {
        var path = TryResolveThumbnailPath(libraryRoot, image);
        if (path == null) return null;

        try
        {
            return new Bitmap(path);
        }
        catch (Exception e)
        {
            Console.WriteLine("Thumbnail loading failed: " + e.Message);
            return null;
        }
    }
    
    /// <summary>
    /// Resolves a thumbnail path for the given image using the service’s lookup
    /// order. First checks the cached thumbnail under the library’s sidecar folder.
    /// If none exists, attempts to locate a sibling preview image next to the scene
    /// file. Returns <c>null</c> if no suitable file is found.
    /// </summary>
    /// <param name="libraryRoot">The library root to search under.</param>
    /// <param name="image">The image node whose thumbnail path is being resolved.</param>
    /// <returns>The resolved thumbnail path, or <c>null</c> if none exists.</returns>
    private static string? TryResolveThumbnailPath(string libraryRoot, ImageNode image)
    {
        if (string.IsNullOrEmpty(image.ImageId)) 
            return null;

        var cached = LibraryPaths.ThumbnailPath(libraryRoot, image.ImageId);
        return File.Exists(cached) ? cached : FindSiblingPreview(libraryRoot, image);
    }
    
    /// <summary>
    /// Attempts to locate a pre‑existing RGB preview image stored alongside the
    /// source scene file. First looks for an image whose basename matches the
    /// scene file (e.g. <c>raw.jpg</c>, <c>raw.png</c>). If none is found, expands
    /// the search to any file with an acceptable preview extension
    /// (<c>.jpg</c>, <c>.jpeg</c>, <c>.png</c>, <c>.bmp</c>). Returns <c>null</c>
    /// if no suitable preview image is found.
    /// </summary>
    /// <param name="libraryRoot">The library root to search under.</param>
    /// <param name="image">The image node whose preview is being resolved.</param>
    /// <returns>The resolved preview image path, or <c>null</c> if none exists.</returns>
    private static string? FindSiblingPreview(string libraryRoot, ImageNode image)
    {
        if (string.IsNullOrEmpty(image.CurrentRelPath)) return null;

        var absPath = Path.Combine(libraryRoot, image.CurrentRelPath.Replace('/', Path.DirectorySeparatorChar));
        var dir = Path.GetDirectoryName(absPath);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        
        var baseName = Path.GetFileNameWithoutExtension(image.SceneFileName);
        foreach (var ext in PreviewExtensions)
        {
            var matches = Directory.GetFiles(dir, baseName + "*")
                .Where(f => !Path.GetFileName(f).StartsWith("._") &&
                            string.Equals(Path.GetExtension(f), ext, StringComparison.OrdinalIgnoreCase))
                .ToArray();
            if (matches.Length == 1) return matches[0];
        }

        foreach (var ext in PreviewExtensions)
        {
            var matches = Directory.GetFiles(dir, "*" + ext)
                .Where(f => !Path.GetFileName(f).StartsWith("._"))
                .ToArray();
            if (matches.Length == 1)
                return matches[0];
        }
        
        return null;
    }

    /// <summary>
    /// Saves a downscaled thumbnail for the given image. If the source bitmap is
    /// already smaller than the maximum edge size, it is saved as-is. Otherwise,
    /// it is proportionally resized so that its longest edge equals <see cref="MaxEdge"/>.
    /// Overwrites any existing cached thumbnail.
    /// </summary>
    /// <param name="libraryRoot">The root of the library to save in.</param>
    /// <param name="imageId">The image ID to give the saved thumbnail.</param>
    /// <param name="source">The bitmap to downscale and save as a thumbnail.</param>
    public static void TrySaveFromBitmap(string libraryRoot, string imageId, Bitmap source)
    {
        if (string.IsNullOrEmpty(imageId)) return;

        try
        {
            LibraryPaths.EnsureSidecarExists(libraryRoot);
            var outPath = LibraryPaths.ThumbnailPath(libraryRoot, imageId);

            var width = source.PixelSize.Width;
            var height = source.PixelSize.Height;
            var scale = (double)MaxEdge / Math.Max(width, height);
            
            if (scale >= 1.0)
            {
                source.Save(outPath);
                return;
            }

            var newW = (int)Math.Round(width * scale);
            var newH = (int)Math.Round(height * scale);
            
            using var scaled = source.CreateScaledBitmap(new PixelSize(newW, newH));
            scaled.Save(outPath);
        }
        catch (Exception e)
        {
            // Silent fail: Skip thumbnail creation
            Console.WriteLine("Thumbnail creation failed: " + e.Message);
        }
    }
}