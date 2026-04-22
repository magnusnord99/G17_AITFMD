using System;
using Avalonia;
using Avalonia.Media.Imaging;

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

    /// <summary>
    /// Attempts to load a cached thumbnail for the given image ID.
    /// Returns <c>null</c> if the thumbnail does not exist or cannot be read.
    /// </summary>
    /// <param name="libraryRoot">The root of the library to look in.</param>
    /// <param name="imageId">The ID of the image to find.</param>
    /// <returns>Returns the thumbnail bitmap if found; otherwise null</returns>
    public static Bitmap? TryLoadFromId(string libraryRoot, string imageId)
    {
        if (string.IsNullOrEmpty(imageId)) return null;
        var path = LibraryPaths.ThumbnailPath(libraryRoot, imageId);
        {
            try
            {
                return new Bitmap(path);
            }
            catch
            {
                return null;
            }
        }
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