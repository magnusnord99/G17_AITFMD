using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SpectralAssist.Models;

namespace SpectralAssist.Services.Inference;

/// <summary>
/// ONNX Runtime classifier for 3D-CNN models with shape <c>[1, 1, C, H, W]</c> (NCDHW).
/// Uses IOBinding to keep tensors on GPU between patch inferences, eliminating per-patch
/// CPU ↔ GPU copies. Falls back to CPU-side Run() when no GPU execution provider is active.
/// </summary>
public class Onnx3DCnnClassifier : IClassifier, IDisposable
{
    private ModelPackage? _package;
    private ExecutionProvider _executionProvider;
    private string? _inputName;
    private string? _outputName;

    public void SetModel(ModelPackage package)
    {
        _package = package;
        ValidateManifest(package.Manifest);
        _executionProvider = _package.ActiveProvider;
        _inputName = _package.Session.InputNames[0];
        _outputName = _package.Session.OutputNames[0];
    }

    public Task<ClassificationResult> ClassifyImageAsync(
        HsiCube cube, 
        bool[]? tissueMask = null,
        int? strideOverride = null, 
        IProgress<(int Done, int Total)>? progress = null,
        CancellationToken ct = default)
    {
        if (_package == null)
            throw new InvalidOperationException("No model loaded");

        var spec = _package.Manifest.InputSpec;
        var grid = new PatchGrid(cube, spec, strideOverride);
        
        if (cube.Bands != grid.Bands)
            throw new InvalidOperationException(
                $"Model expects {grid.Bands} spectral bands, cube has {cube.Bands}.");
        
        ct.ThrowIfCancellationRequested();
        
        //ToDo: Check execution provider logic vs _useGPU? IOBinding vs CPU?
        var predictions = _executionProvider == ExecutionProvider.Cuda
            ? ClassifyAllPatchesWithIOBinding(cube, grid, tissueMask, progress, ct)
            : ClassifyAllPatchesCpu(cube, grid, tissueMask, progress, ct);
        
        return Task.FromResult(new ClassificationResult
        {
            Predictions = predictions,
            ImageWidth = cube.Samples,
            ImageHeight = cube.Lines,
            PatchW = grid.PatchW,
            PatchH = grid.PatchH,
            StrideH = grid.StrideH,
            StrideW = grid.StrideW,
            Classes = _package.Manifest.OutputSpec.Classes,
            ModelName = _package.Manifest.Metadata.Name,
            TotalPossible = grid.TotalPatches,
            Evaluated = predictions.Count,
            Skipped = grid.TotalPatches - predictions.Count,
            ExecutionProvider = _executionProvider.ToString(),
        });
    }
    
    /// <summary>
    /// Precomputes all sliding-window geometry, including tile counts, strides, and patch dimensions .
    /// </summary>
    private readonly struct PatchGrid
    {
        public readonly int Bands, PatchH, PatchW, StrideH, StrideW, TilesY, TilesX, TotalPatches, PatchSize;
        public readonly long[] InputShape;

        public PatchGrid(HsiCube cube, InputSpec spec, int? strideOverride)
        {
            Bands = spec.SpectralBands;
            PatchH = spec.SpatialPatchSize[0];
            PatchW = spec.SpatialPatchSize[1];
            StrideH = strideOverride ?? spec.Stride[0];
            StrideW = strideOverride ?? spec.Stride[1];
            PatchSize = Bands * PatchH * PatchW;
            InputShape = [1, 1, Bands, PatchH, PatchW];

            // Ceiling division ensures the boundary tile is included (clamped in the loop)
            TilesY = Math.Max(1, (int)Math.Ceiling((cube.Lines - PatchH) / (double)StrideH) + 1);
            TilesX = Math.Max(1, (int)Math.Ceiling((cube.Samples - PatchW) / (double)StrideW) + 1);
            TotalPatches = TilesY * TilesX;
        }
    }



    /// <summary>
    /// Uses IOBinding to keep the output tensor on GPU between patches.
    /// Input is pinned CPU memory (OrtMemoryInfo.DefaultInstance) which ORT copies to GPU once.
    /// Output is bound to GPU memory, no per-patch GPU → CPU copy until we explicitly fetch results.
    /// </summary>
    private List<PatchPrediction> ClassifyAllPatchesWithIOBinding(
        HsiCube cube, PatchGrid grid, bool[]? tissueMask,
        IProgress<(int Done, int Total)>? progress, CancellationToken ct)
    {
        var session = _package!.Session;
        var predictions = new List<PatchPrediction>();
        var done = 0;

        // Rent a reusable buffer for patch extraction (avoids per-patch allocation)
        var patchBuffer = ArrayPool<float>.Shared.Rent(grid.PatchSize);
        try
        {
            for (var ty = 0; ty < grid.TilesY; ty++)
            {
                var y = Math.Min(ty * grid.StrideH, cube.Lines - grid.PatchH);
                for (var tx = 0; tx < grid.TilesX; tx++)
                {
                    ct.ThrowIfCancellationRequested();
                    var x = Math.Min(tx * grid.StrideW, cube.Samples - grid.PatchW);

                    // Skip patches that don't overlap any tissue
                    if (!PatchOverlapsTissue(tissueMask, cube.Samples, x, y, grid.PatchW, grid.PatchH))
                    {
                        progress?.Report((++done, grid.TotalPatches));
                        continue;
                    }
                    
                    // Extract patch into reusable buffer (no allocation)
                    ExtractPatchInto(cube, x, y, grid.PatchW, grid.PatchH, patchBuffer);

                    // Create OrtValue from managed Memory<float> with length equal to patch size
                    using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        patchBuffer.AsMemory(0, grid.PatchSize),
                        grid.InputShape);
                    
                    // Bind input (CPU) and output (GPU), and run inference.
                    using var binding = session.CreateIoBinding();
                    binding.BindInput(_inputName!, inputOrtValue);
                    binding.BindOutputToDevice(_outputName!, OrtMemoryInfo.DefaultInstance);
                    session.RunWithBinding(new RunOptions(), binding);

                    // Fetch output
                    using var outputs = binding.GetOutputValues();
                    var logits = outputs[0].GetTensorDataAsSpan<float>().ToArray();
                    var probabilities = Softmax(logits);

                    predictions.Add(new PatchPrediction
                    {
                        X = x,
                        Y = y,
                        Probabilities = probabilities,
                        Logits = logits
                    });
                    progress?.Report((++done, grid.TotalPatches));
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
        HsiCube cube, PatchGrid grid, bool[]? tissueMask,
        IProgress<(int Done, int Total)>? progress, CancellationToken ct)
    {
        var session = _package!.Session;
        var predictions = new List<PatchPrediction>();
        var done = 0;

        var patchBuffer = ArrayPool<float>.Shared.Rent(grid.PatchSize);
        try
        {
            for (var ty = 0; ty < grid.TilesY; ty++)
            {
                var y = Math.Min(ty * grid.StrideH, cube.Lines - grid.PatchH);
                for (var tx = 0; tx < grid.TilesX; tx++)
                {
                    ct.ThrowIfCancellationRequested();
                    var x = Math.Min(tx * grid.StrideW, cube.Samples - grid.PatchW);

                    // Skip patches that don't overlap any tissue
                    if (!PatchOverlapsTissue(tissueMask, cube.Samples, x, y, grid.PatchW, grid.PatchH))
                    {
                        progress?.Report((++done, grid.TotalPatches));
                        continue;
                    }

                    ExtractPatchInto(cube, x, y, grid.PatchW, grid.PatchH, patchBuffer);

                    // DenseTensor wraps the memory segment (exact length, not pooled capacity)
                    var tensor = new DenseTensor<float>(
                        patchBuffer.AsMemory(0, grid.PatchSize),
                        new[] { 1, 1, grid.Bands, grid.PatchH, grid.PatchW });

                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName!, tensor),
                    };

                    using var results = session.Run(inputs);
                    var logits = results[0].AsEnumerable<float>().ToArray();
                    var probabilities = Softmax(logits);

                    predictions.Add(new PatchPrediction
                    {
                        X = x,
                        Y = y,
                        Probabilities = probabilities,
                        Logits = logits
                    });
                    progress?.Report((++done, grid.TotalPatches));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(patchBuffer);
        }

        return predictions;
    }

    
    // -- Helpers -- //
    
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
        var inputSpec = manifest.InputSpec;

        if (inputSpec.SpatialPatchSize.Count < 2)
            throw new InvalidOperationException(
                "input_spec.spatial_patch_size must have at least 2 elements.");

        if (inputSpec.InputRank != 5)
            throw new InvalidOperationException(
                $"Onnx3DCnnClassifier requires input_rank = 5 (NCDHW), got {inputSpec.InputRank}.");

        if (inputSpec.InputShape.Count == 5 && inputSpec.InputShape[2] != inputSpec.SpectralBands)
            throw new InvalidOperationException(
                $"input_shape[2] ({inputSpec.InputShape[2]}) does not match spectral_bands ({inputSpec.SpectralBands}).");
    }

    public void Dispose()
    {
        _package = null;
        GC.SuppressFinalize(this);
    }
}