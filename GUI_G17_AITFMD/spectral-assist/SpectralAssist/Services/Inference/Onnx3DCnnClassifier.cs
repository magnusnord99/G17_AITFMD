using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

/// <summary>
/// ONNX Runtime classifier for 3D-CNN models with shape <c>[1, 1, C, H, W]</c> (NCDHW).
/// Uses IOBinding to keep tensors on GPU between patch inferences, eliminating per-patch
/// CPU↔GPU copies. Falls back to CPU-side Run() when no GPU execution provider is active.
/// </summary>
public class Onnx3DCnnClassifier : IClassifier, IDisposable
{
    private ModelPackage? _package;

    /// <summary>Cached input/output names from the ONNX model metadata.</summary>
    private string? _inputName;

    private string? _outputName;

    /// <summary>Whether the session has a GPU execution provider (CUDA/DirectML/etc).</summary>
    private bool _useGpu;

    private ExecutionProvider _executionProvider;

    public ModelManifest? Manifest => _package?.Manifest;

    public void SetModel(ModelPackage package)
    {
        _package?.Dispose();
        _package = package;
        ValidateManifest(_package.Manifest);

        // Cache input/output tensor names from model metadata
        _inputName = _package.Session.InputNames[0];
        _outputName = _package.Session.OutputNames[0];

        // Detect whether the session was created with a GPU provider
        _useGpu = package.UseGpu;
        _executionProvider = package.ActiveProvider;
    }

    public Task<ClassificationResult> ClassifyImageAsync(
        HsiCube cube, bool[]? tissueMask = null, IProgress<(int Done, int Total)>? progress = null)
    {
        if (_package == null)
            throw new InvalidOperationException("No model loaded");

        var spec = _package.Manifest.InputSpec;
        var patchH = spec.SpatialPatchSize[0];
        var patchW = spec.SpatialPatchSize[1];
        var bands = spec.SpectralBands;

        if (cube.Bands != bands)
            throw new InvalidOperationException(
                $"Model expects {bands} spectral bands, cube has {cube.Bands}.");

        var predictions = _useGpu
            ? ClassifyAllPatchesWithIOBinding(cube, bands, patchH, patchW, tissueMask, progress)
            : ClassifyAllPatchesCpu(cube, bands, patchH, patchW, tissueMask, progress);

        var tilesY = (cube.Lines + patchH - 1) / patchH;
        var tilesX = (cube.Samples + patchW - 1) / patchW;

        return Task.FromResult(new ClassificationResult
        {
            Predictions = predictions,
            ImageWidth = cube.Samples,
            ImageHeight = cube.Lines,
            PatchW = patchW,
            PatchH = patchH,
            Classes = _package.Manifest.OutputSpec.Classes,
            ModelName = _package.Manifest.Metadata.Name,
            TotalPossible = tilesY * tilesX,
            Evaluated = predictions.Count,
            Skipped = tilesY * tilesX - predictions.Count,
            ExecutionProvider = _executionProvider.ToString(),
        });
    }


    /// <summary>
    /// Uses IOBinding to keep the output tensor on GPU between patches.
    /// Input is pinned CPU memory (OrtMemoryInfo.DefaultInstance) which ORT copies to GPU once.
    /// Output is bound to GPU memory, no per-patch GPU→CPU copy until we explicitly fetch results.
    /// </summary>
    private List<PatchPrediction> ClassifyAllPatchesWithIOBinding(
        HsiCube cube, int bands, int patchH, int patchW,
        bool[]? tissueMask = null,
        IProgress<(int Done, int Total)>? progress = null)
    {
        var session = _package!.Session;
        var predictions = new List<PatchPrediction>();
        var patchSize = bands * patchH * patchW;
        var inputShape = new long[] { 1, 1, bands, patchH, patchW };
        var tilesY = (cube.Lines + patchH - 1) / patchH;
        var tilesX = (cube.Samples + patchW - 1) / patchW;
        var totalPatches = tilesY * tilesX;
        var done = 0;

        // Rent a reusable buffer for patch extraction (avoids per-patch allocation)
        var patchBuffer = ArrayPool<float>.Shared.Rent(patchSize);
        try
        {
            for (var y = 0; y < cube.Lines; y += patchH)
            {
                var startY = Math.Min(y, cube.Lines - patchH);
                for (var x = 0; x < cube.Samples; x += patchW)
                {
                    var startX = Math.Min(x, cube.Samples - patchW);

                    // Skip patches that don't overlap any tissue
                    if (!PatchOverlapsTissue(tissueMask, cube.Samples, startX, startY, patchW, patchH))
                    {
                        progress?.Report((++done, totalPatches));
                        continue;
                    }

                    // Extract patch into reusable buffer (no allocation)
                    ExtractPatchInto(cube, startX, startY, patchW, patchH, patchBuffer);

                    // Create OrtValue from managed Memory<float> — ORT pins it internally during inference
                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        patchBuffer.AsMemory(0, patchSize),
                        inputShape);

                    using var binding = session.CreateIoBinding();

                    // Bind input (CPU) and output (GPU), and run inference.
                    binding.BindInput(_inputName!, inputOrtValue);
                    binding.BindOutputToDevice(_outputName!, OrtMemoryInfo.DefaultInstance);
                    session.RunWithBinding(new RunOptions(), binding);

                    // Fetch output
                    using var outputs = binding.GetOutputValues();
                    var logits = outputs[0].GetTensorDataAsSpan<float>().ToArray();
                    var probs = Softmax(logits);

                    predictions.Add(new PatchPrediction
                    {
                        X = startX,
                        Y = startY,
                        Probabilities = probs,
                    });

                    progress?.Report((++done, totalPatches));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(patchBuffer);
        }

        return predictions;
    }


    /// <summary>
    /// CPU fallback using standard session.Run().
    /// Still uses pooled buffers to reduce GC pressure.
    /// </summary>
    private List<PatchPrediction> ClassifyAllPatchesCpu(
        HsiCube cube, int bands, int patchH, int patchW,
        bool[]? tissueMask = null,
        IProgress<(int Done, int Total)>? progress = null)
    {
        var session = _package!.Session;
        var predictions = new List<PatchPrediction>();
        var patchSize = bands * patchH * patchW;
        var tilesY = (cube.Lines + patchH - 1) / patchH;
        var tilesX = (cube.Samples + patchW - 1) / patchW;
        var totalPatches = tilesY * tilesX;
        var done = 0;

        var patchBuffer = ArrayPool<float>.Shared.Rent(patchSize);
        try
        {
            for (var y = 0; y < cube.Lines; y += patchH)
            {
                var startY = Math.Min(y, cube.Lines - patchH);
                for (var x = 0; x < cube.Samples; x += patchW)
                {
                    var startX = Math.Min(x, cube.Samples - patchW);

                    // Skip patches that don't overlap any tissue
                    if (!PatchOverlapsTissue(tissueMask, cube.Samples, startX, startY, patchW, patchH))
                    {
                        progress?.Report((++done, totalPatches));
                        continue;
                    }

                    ExtractPatchInto(cube, startX, startY, patchW, patchH, patchBuffer);

                    // DenseTensor wraps the memory segment (exact length, not pooled capacity)
                    var tensor = new DenseTensor<float>(
                        patchBuffer.AsMemory(0, patchSize),
                        new[] { 1, 1, bands, patchH, patchW });

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName!, tensor),
                    };

                    using var results = session.Run(inputs);
                    var logits = results[0].AsEnumerable<float>().ToArray();
                    var probs = Softmax(logits);

                    predictions.Add(new PatchPrediction
                    {
                        X = startX,
                        Y = startY,
                        Probabilities = probs,
                    });

                    progress?.Report((++done, totalPatches));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(patchBuffer);
        }

        return predictions;
    }

    /// <summary>
    /// Extracts a BSQ patch from the cube into a pre-allocated buffer.
    /// Same layout as <see cref="HsiCube.ExtractPatch"/> but avoids allocation.
    /// </summary>
    private static void ExtractPatchInto(
        HsiCube cube, int startX, int startY, int patchW, int patchH, float[] dest)
    {
        for (var b = 0; b < cube.Bands; b++)
        {
            var band = cube.GetBand(b);
            var destOffset = b * patchH * patchW;
            for (var row = 0; row < patchH; row++)
            {
                band.Slice((startY + row) * cube.Samples + startX, patchW)
                    .CopyTo(dest.AsSpan(destOffset + row * patchW, patchW));
            }
        }
    }

    /// <summary>
    /// Returns true if any pixel in the patch region is marked as tissue (true) in the mask.
    /// If mask is null, always returns true (classify everything).
    /// </summary>
    private static bool PatchOverlapsTissue(
        bool[]? mask, int imageWidth, int startX, int startY, int patchW, int patchH)
    {
        if (mask == null) return true;
        for (var row = startY; row < startY + patchH; row++)
        for (var col = startX; col < startX + patchW; col++)
            if (mask[row * imageWidth + col])
                return true;
        return false;
    }

    private static float[] Softmax(float[] logits)
    {
        if (logits.Length == 0)
            return logits;
        var max = logits.Max();
        var exp = new float[logits.Length];
        var sum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            sum += exp[i];
        }

        if (sum <= 0f)
            return logits;
        for (var i = 0; i < logits.Length; i++)
            exp[i] /= sum;
        return exp;
    }

    /// <summary>
    /// Validates that the manifest describes a 5D (NCDHW) model compatible
    /// with this 3D-CNN classifier. Called once when <see cref="SetModel"/> is invoked.
    /// </summary>
    private static void ValidateManifest(ModelManifest manifest)
    {
        var spec = manifest.InputSpec;

        if (spec.SpatialPatchSize.Count < 2)
            throw new InvalidOperationException(
                "input_spec.spatial_patch_size must have at least 2 elements.");

        if (spec.InputRank != 5)
            throw new InvalidOperationException(
                $"Onnx3DCnnClassifier requires input_rank = 5 (NCDHW), got {spec.InputRank}.");

        if (spec.InputShape.Count == 5 && spec.InputShape[2] != spec.SpectralBands)
            throw new InvalidOperationException(
                $"input_shape[2] ({spec.InputShape[2]}) does not match spectral_bands ({spec.SpectralBands}).");
    }

    public void Dispose()
    {
        _package?.Dispose();
        _package = null;
        GC.SuppressFinalize(this);
    }
}