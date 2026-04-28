using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using SpectralAssist.Extensions;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Library;

/// <summary>
/// Manages the currently open library. Handles loading, rescanning, and saving
/// the manifest, and coordinates all filesystem I/O related to the library.
/// This is the main entry point for interacting with a dataset.
/// </summary>
public class LibraryManager
{
    private readonly SemaphoreSlim _manifestLock = new(1, 1);

    public string? Root { get; private set; }
    public LibraryManifest? Manifest { get; private set; }
    public bool IsOpen => Root != null && Manifest != null;

    public event Action<ImageNode>? ImageUpdated;
    public void NotifyImageUpdated(ImageNode node) => ImageUpdated?.Invoke(node);

    /// <summary>
    /// Opens a library folder, loads or generates its manifest, and initializes the manager's state.
    /// If a manifest already exists, its image IDs and notes are preserved during the scan.
    /// </summary>
    /// <param name="libraryRoot">Path to the library folder.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task OpenAsync(string libraryRoot, CancellationToken ct = default)
    {
        var existingManifest = await TryReadManifestAsync(libraryRoot, ct);
        var freshManifest = await LibraryScanner.ScanAsync(libraryRoot, existingManifest, ct);

        await _manifestLock.WaitAsync(ct);
        try
        {
            await WriteManifestAsync(libraryRoot, freshManifest, ct);
            Root = libraryRoot;
            Manifest = freshManifest;
        }
        finally
        {
            _manifestLock.Release();
        }
    }


    /// <summary>
    /// Re-scans the currently open library and updates the manifest while
    /// preserving existing image IDs and notes.
    /// </summary>
    /// <param name="ct">Cancellation token.</param>
    public async Task RescanAsync(CancellationToken ct = default)
    {
        if (Root == null) return;
        var freshManifest = await LibraryScanner.ScanAsync(Root, Manifest, ct);

        await _manifestLock.WaitAsync(ct);
        try
        {
            await WriteManifestAsync(Root, freshManifest, ct);
            Manifest = freshManifest;
        }
        finally
        {
            _manifestLock.Release();
        }
    }

    /// <summary>
    /// Closes the currently open library and clears all associated states.
    /// </summary>
    public void Close()
    {
        Root = null;
        Manifest = null;
    }

    /// <summary>
    /// Looks up an image by its ID in the currently loaded manifest.
    /// Returns <c>null</c> if the image is not found or no library is open.
    /// </summary>
    /// <param name="imageId">The ID of the <c>ImageNode</c> to find.</param>
    /// <returns>Returns found <c>ImageNode</c> if found; otherwise <c>null</c></returns>
    private ImageNode? FindImage(string imageId)
    {
        if (Manifest == null) return null;
        return LibraryScanner.FlattenImages(Manifest.Folders)
            .FirstOrDefault(i => i.ImageId == imageId);
    }


    /// <summary>
    /// Saves a classification run for the specified image, generating a new run ID
    /// if needed, writing the run report to disk, and updating the manifest with a
    /// summarized entry. Preserves existing run history for the image.
    /// </summary>
    /// <param name="imageId">The ID of the image the run belongs to.</param>
    /// <param name="report">The full classification report to store.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A summary of the saved run.</returns>
    /// <exception cref="InvalidOperationException">Thrown if no library is open.</exception>
    public async Task<RunSummary> SaveRunAsync(string imageId, ClassificationReport report,
        CancellationToken ct = default)
    {
        if (Root == null || Manifest == null)
            throw new InvalidOperationException("No library open.");

        LibraryPaths.EnsureSidecarExists(Root);
        var runId = string.IsNullOrEmpty(report.RunId) ? NewRunId(report.ModelDisplayName) : report.RunId;

        var toSave = new ClassificationReport
        {
            RunId = runId,
            ImageId = imageId,
            CompletedAt = report.CompletedAt == default ? DateTime.UtcNow : report.CompletedAt,

            ImageWidth = report.ImageWidth,
            ImageHeight = report.ImageHeight,
            PatchH = report.PatchH,
            PatchW = report.PatchW,
            StrideH = report.StrideH,
            StrideW = report.StrideW,
            TotalPatches = report.TotalPatches,
            EvaluatedPatches = report.EvaluatedPatches,
            SkippedPatches = report.SkippedPatches,
            ExecutionProvider = report.ExecutionProvider,

            Classes = report.Classes,
            Statistics = report.Statistics.Count > 0 ? report.Statistics : StatisticsCalculator.Compute(report),
            Predictions = report.Predictions,

            ModelMetadata = report.ModelMetadata,
            ModelTraining = report.ModelTraining,
            ModelSummary = report.ModelSummary,
            PreprocessingSteps = report.PreprocessingSteps,
            SpectralReducer = report.SpectralReducer,
        };

        Directory.CreateDirectory(LibraryPaths.RunFolder(Root, imageId));
        var runPath = LibraryPaths.RunPath(Root, imageId, runId);
        await WriteJsonSafelyAsync(runPath, toSave, ct);

        var (positiveName, pct50, pct80) = StatisticsCalculator.PickPositiveClass(toSave.Statistics);
        var summary = new RunSummary
        {
            RunId = runId,
            ModelDisplayName = toSave.ModelDisplayName,
            CompletedAt = toSave.CompletedAt,
            PositiveClassName = positiveName,
            PositiveClassPercentAbove50 = pct50,
            PositiveClassPercentAbove80 = pct80,
        };

        await _manifestLock.WaitAsync(ct);
        try
        {
            var image = FindImage(imageId);
            if (image != null)
            {
                image.Runs.RemoveAll(r => r.RunId == runId);
                image.Runs.Add(summary);
                image.Runs.Sort((a, b) => b.CompletedAt.CompareTo(a.CompletedAt));
                ImageUpdated?.Invoke(image);
            }

            await WriteManifestAsync(Root, Manifest, ct);
        }
        finally
        {
            _manifestLock.Release();
        }

        return summary;
    }


    /// <summary>
    /// Loads a previously saved classification run for the given image and run ID.
    /// Returns <c>null</c> if the run does not exist or no library is open.
    /// </summary>
    /// <param name="imageId">The ID of the image the run belongs to.</param>
    /// <param name="runId">The ID of the run to load.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The deserialized <see cref="ClassificationReport"/> or <c>null</c>.</returns>
    public async Task<ClassificationReport?> LoadRunAsync(string imageId, string runId, CancellationToken ct = default)
    {
        if (Root == null) return null;
        var path = LibraryPaths.RunPath(Root, imageId, runId);
        if (!File.Exists(path)) return null;

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<ClassificationReport>(stream, LibraryJson.Options, ct);
    }


    /// <summary>
    /// Deletes a stored inference run for the specified image. Removes the run file,
    /// updates the manifest, and deletes the per‑image run folder under
    /// <c>.spectral-assist/runs/</c> if it becomes empty. The image’s data folder
    /// is never modified.
    /// </summary>
    /// <param name="imageId">The ID of the image the run belongs to.</param>
    /// <param name="runId">The ID of the run to delete.</param>
    /// <param name="ct">Cancellation token.</param>
    public async Task DeleteRunAsync(string imageId, string runId, CancellationToken ct = default)
    {
        if (Root == null || Manifest == null) return;

        var path = LibraryPaths.RunPath(Root, imageId, runId);
        if (File.Exists(path)) File.Delete(path);

        await _manifestLock.WaitAsync(ct);
        try
        {
            var image = FindImage(imageId);
            image?.Runs.RemoveAll(r => r.RunId == runId);
            ImageUpdated?.Invoke(image!);
            await WriteManifestAsync(Root, Manifest, ct);
        }
        finally
        {
            _manifestLock.Release();
        }

        var runFolder = LibraryPaths.RunFolder(Root, imageId);
        if (Directory.Exists(runFolder) && !Directory.EnumerateFileSystemEntries(runFolder).Any())
            Directory.Delete(runFolder);
    }

    /// <summary>
    /// Attempts to read and deserialize the manifest from the given library root.
    /// Returns <c>null</c> if the manifest file does not exist or cannot be parsed.
    /// </summary>
    /// <param name="libraryRoot">Path to the library folder.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The loaded <see cref="LibraryManifest"/> or <c>null</c>.</returns>
    private static async Task<LibraryManifest?> TryReadManifestAsync(string libraryRoot, CancellationToken ct = default)
    {
        var path = LibraryPaths.ManifestPath(libraryRoot);
        if (!File.Exists(path)) return null;

        try
        {
            await using var stream = File.OpenRead(path);
            return await JsonSerializer.DeserializeAsync<LibraryManifest>(stream, LibraryJson.Options, ct);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Writes the manifest to the library’s sidecar folder using a safe write‑and‑replace strategy.
    /// Updates the manifest’s <see cref="LibraryManifest.LastScanned"/> timestamp before saving.
    /// </summary>
    /// <param name="libraryRoot">Path to the library folder.</param>
    /// <param name="manifest">The manifest to write.</param>
    /// <param name="ct">Cancellation token.</param>
    private static async Task WriteManifestAsync(string libraryRoot, LibraryManifest manifest,
        CancellationToken ct = default)
    {
        LibraryPaths.EnsureSidecarExists(libraryRoot);
        manifest.LastScanned = DateTime.UtcNow;

        var path = LibraryPaths.ManifestPath(libraryRoot);
        var tempPath = path + ".tmp";

        await using (var filestream = File.Create(tempPath))
            await JsonSerializer.SerializeAsync(filestream, manifest, LibraryJson.Options, ct);

        if (File.Exists(path))
            File.Replace(tempPath, path, path + ".bak");
        else
            File.Move(tempPath, path);
    }

    /// <summary>
    /// Writes a JSON file safely by serializing to a temporary file and then
    /// replacing or moving it into place. Ensures the target directory exists.
    /// </summary>
    /// <param name="path">Destination file path.</param>
    /// <param name="value">The object to serialize.</param>
    /// <param name="ct">Cancellation token.</param>
    private static async Task WriteJsonSafelyAsync<T>(string path, T value, CancellationToken ct = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var tempPath = path + ".tmp";
        await using (var filestream = File.Create(tempPath))
        {
            await JsonSerializer.SerializeAsync(filestream, value, LibraryJson.Options, ct);
        }

        if (File.Exists(path))
            File.Replace(tempPath, path, null);

        else File.Move(tempPath, path);
    }

    /// <summary>
    /// Generates a unique run ID consisting of a UTC timestamp, a sanitized model
    /// name, and a short GUID suffix. Suitable for use as a filename.
    /// </summary>
    /// <param name="modelName">The model name to embed in the ID.</param>
    /// <returns>A unique run identifier.</returns>
    private static string NewRunId(string modelName)
    {
        var timestamp = DateTime.UtcNow.ToString("yyyyMMddTHHmmssZ", CultureInfo.InvariantCulture);
        var sanitizedModelName = string.Join("_", modelName.Split(Path.GetInvalidFileNameChars()));
        var guid = Guid.NewGuid().ToString("N")[..8];
        return $"{timestamp}__{sanitizedModelName}__{guid}";
    }
}