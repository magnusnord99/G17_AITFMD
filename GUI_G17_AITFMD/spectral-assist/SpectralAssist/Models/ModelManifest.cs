using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json.Serialization;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.Models;

/// <summary>
/// Root DTO for the model package manifest, <c>manifest.json</c>.
/// Mirrors the JSON contract between the Python export pipeline and the C# application.
/// The added <see cref="Id"/> and <see cref="DirectoryPath"/> properties are added
/// by <see cref="Services.ModelPackageService"/> for runtime use.
/// </summary>
public class ModelManifest
{
    /// <summary>Unique identifier within ModelPackages/.</summary>
    [JsonIgnore] public string Id { get; set; } = string.Empty;

    /// <summary>Absolute path to the model package directory.</summary>
    [JsonIgnore] public string DirectoryPath { get; set; } = string.Empty;
    
    // --- Display Helpers --- //
    
    [JsonIgnore] public string DisplayName =>
        string.IsNullOrWhiteSpace(Metadata.Name) ? Id : $"{Metadata.Name} v{Metadata.Version}";

    [JsonIgnore] public string ClassesDisplay =>
        OutputSpec.Classes.Count > 0 ? string.Join(", ", OutputSpec.Classes) : "";

    [JsonIgnore] public string PatchSizeDisplay =>
        InputSpec.SpatialPatchSize.Count >= 2
            ? $"{InputSpec.SpatialPatchSize[0]} × {InputSpec.SpatialPatchSize[1]}"
            : "";
    
    [JsonIgnore] public string? ArchitectureDiagramPath =>
        !string.IsNullOrEmpty(Artifacts.ArchitectureDiagram) 
            ? Path.Combine(DirectoryPath, Artifacts.ArchitectureDiagram) 
            : null;
    
    public override string ToString() => DisplayName;
    
    
    // --- JSON Properties --- //
    
    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; set; } = string.Empty;

    [JsonPropertyName("metadata")]
    public ManifestMetadata Metadata { get; set; } = new();

    [JsonPropertyName("pipeline")]
    public PipelineInfo Pipeline { get; set; } = new();

    [JsonPropertyName("input_spec")]
    public InputSpec InputSpec { get; set; } = new();

    [JsonPropertyName("output_spec")]
    public OutputSpec OutputSpec { get; set; } = new();

    [JsonPropertyName("training")]
    public TrainingInfo Training { get; set; } = new();

    [JsonPropertyName("artifacts")]
    public ArtifactPaths Artifacts { get; set; } = new();

    [JsonPropertyName("validation")]
    public ValidationInfo Validation { get; set; } = new();
}

/// <summary>Display metadata: name, version, creation date, author, and description.</summary>
public class ManifestMetadata
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;

    [JsonPropertyName("created")]
    public string Created { get; set; } = string.Empty;

    [JsonPropertyName("author")]
    public string Author { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;
}

/// <summary>The three stages of the inference pipeline: preprocess → reduce → model.</summary>
public class PipelineInfo
{
    [JsonPropertyName("preprocessing")]
    public PreprocessingInfo Preprocessing { get; set; } = new();

    [JsonPropertyName("spectral_reducer")]
    public SpectralReducerInfo SpectralReducer { get; set; } = new();

    [JsonPropertyName("model")]
    public ModelInfo Model { get; set; } = new();
}

/// <summary>Ordered preprocessing steps and their parameters.
/// Executed by PreprocessingService to replicate the Python training pipeline.</summary>
public class PreprocessingInfo
{
    [JsonPropertyName("steps")]
    public List<string> Steps { get; set; } = [];

    [JsonPropertyName("params")]
    public PreprocessingConfig Params { get; set; } = new();
}

/// <summary>Preprocessing hyperparameters matching the Python training pipeline.
/// Used directly by PreprocessingService to reproduce identical preprocessing.</summary>
public class PreprocessingConfig
{
    [JsonPropertyName("calibration_epsilon")]
    public float? CalibrationEpsilon { get; set; }

    [JsonPropertyName("clip_min")]
    public float? ClipMin { get; set; }

    [JsonPropertyName("clip_max")]
    public float? ClipMax { get; set; }

    [JsonPropertyName("neighbor_average_window")]
    public int? NeighborAverageWindow { get; set; }
    
    [JsonPropertyName("tissue_mask_method")]
    public string? TissueMaskMethod { get; set; }

    [JsonPropertyName("tissue_mask_q_mean")]
    public float? TissueMaskQMean { get; set; }

    [JsonPropertyName("tissue_mask_q_std")]
    public float? TissueMaskQStd { get; set; }

    [JsonPropertyName("tissue_mask_min_object_size")]
    public int? TissueMaskMinObjectSize { get; set; }

    [JsonPropertyName("tissue_mask_min_hole_size")]
    public int? TissueMaskMinHoleSize { get; set; }
    
    [JsonPropertyName("band_reduce_out_bands")]
    public int? BandReduceOutBands { get; set; }

    [JsonPropertyName("band_reduce_strategy")]
    public string? BandReduceStrategy { get; set; }
}

/// <summary>Spectral band reduction stage. May run in C# (band_average)
/// or be embedded in the ONNX graph (PCA, autoencoder).</summary>
public class SpectralReducerInfo
{
    [JsonPropertyName("method")]
    public string Method { get; set; } = string.Empty;

    [JsonPropertyName("embedded_in_onnx")] 
    public bool EmbeddedInOnnx { get; set; }

    /// <summary>Bands entering the reducer (e.g. 826 or 275).
    /// Equals input_spec.spectral_bands when <c>embedded_in_onnx</c> is true.</summary>
    [JsonPropertyName("input_bands")]
    public int? InputBands { get; set; }

    /// <summary>Bands after reduction (e.g. 16).
    /// Equals input_spec.spectral_bands when <c>embedded_in_onnx</c> is false.</summary>
    [JsonPropertyName("output_bands")]
    public int? OutputBands { get; set; }
}

/// <summary>The neural network stage of the pipeline: architecture name, task type,
/// parameter counts, and a breakdown of layer types.</summary>
public class ModelInfo
{
    [JsonPropertyName("architecture")]
    public string Architecture { get; set; } = string.Empty;

    /// <summary>The model's inference task, e.g. "classification" or "segmentation".
    /// Determines how the app interprets and displays the output.</summary>
    [JsonPropertyName("task")]
    public string Task { get; set; } = string.Empty;

    [JsonPropertyName("total_parameters")]
    public long? TotalParameters { get; set; }

    [JsonPropertyName("trainable_parameters")]
    public long? TrainableParameters { get; set; }
    
    [JsonPropertyName("layers")]
    public Dictionary<string, int> Layers { get; set; } = new();

}

/// <summary>The ONNX model's expected input tensor shape, layout, and data type.</summary>
public class InputSpec
{
    [JsonPropertyName("input_rank")]
    public int InputRank { get; set; }

    [JsonPropertyName("tensor_layout")]
    public string TensorLayout { get; set; } = string.Empty;

    [JsonPropertyName("input_shape")]
    public List<int> InputShape { get; set; } = [];

    [JsonPropertyName("spectral_bands")]
    public int SpectralBands { get; set; }

    [JsonPropertyName("spatial_patch_size")]
    public List<int> SpatialPatchSize { get; set; } = [];

    [JsonPropertyName("dtype")]
    public string Dtype { get; set; } = string.Empty;
}

/// <summary>The ONNX model's output: type (logits/softmax), class count, and class names.</summary>
public class OutputSpec
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("num_classes")]
    public int? NumClasses { get; set; }

    [JsonPropertyName("classes")]
    public List<string> Classes { get; set; } = [];
}

/// <summary>Training context: dataset, sample count, epoch count, and evaluation metrics.</summary>
public class TrainingInfo
{
    [JsonPropertyName("dataset")]
    public string? Dataset { get; set; }

    [JsonPropertyName("samples")]
    public int? Samples { get; set; }

    [JsonPropertyName("epochs")]
    public int? Epochs { get; set; }

    [JsonPropertyName("metrics")]
    public TrainingMetrics Metrics { get; set; } = new();
}

/// <summary>Evaluation metrics from the training validation set.</summary>
public class TrainingMetrics
{
    [JsonPropertyName("accuracy")]
    public double? Accuracy { get; set; }

    [JsonPropertyName("precision")]
    public double? Precision { get; set; }

    [JsonPropertyName("recall")]
    public double? Recall { get; set; }

    [JsonPropertyName("f1")]
    public double? F1 { get; set; }
}

/// <summary>File paths within the model package folder: ONNX model,
/// optional architecture diagram, and optional validation artifacts.</summary>
public class ArtifactPaths
{
    [JsonPropertyName("model_onnx")] 
    public string ModelOnnx { get; set; } = string.Empty;

    [JsonPropertyName("architecture_diagram")]
    public string? ArchitectureDiagram { get; set; }
}

/// <summary>Validation information used for the smoke test at import time.</summary>
public class ValidationInfo
{
    [JsonPropertyName("status")] 
    public string Status { get; set; } = string.Empty;
    
    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;

    [JsonPropertyName("roi_dir")]
    public string? RoiDir { get; set; }
    
    [JsonPropertyName("tolerance")]
    public float? Tolerance { get; set; } = 1e-4f;

    [JsonPropertyName("expected_output")]
    public ExpectedOutput? ExpectedOutput { get; set; }
    
    [JsonPropertyName("actual_output")]
    public ActualOutput? ActualOutput { get; set; }
}

/// <summary>Python validation result, written to manifest during export pipeline test.</summary>
public class ExpectedOutput
{
    [JsonPropertyName("logits")]
    public List<float> Logits { get; set; } = [];

    [JsonPropertyName("softmax")] 
    public List<float> Softmax { get; set; } = [];

    [JsonPropertyName("predicted_class")]
    public int? PredictedClass { get; set; }
}

/// <summary>C# validation result, written to manifest after import-time smoke test.</summary>
public class ActualOutput
{
    [JsonPropertyName("logits")]
    public List<float> Logits { get; set; } = [];

    [JsonPropertyName("softmax")] 
    public List<float> Softmax { get; set; } = [];

    [JsonPropertyName("predicted_class")] 
    public int? PredictedClass { get; set; }

    [JsonPropertyName("max_logit_diff")]
    public float? MaxLogitDiff { get; set; }

    [JsonPropertyName("validated_at")]
    public string ValidatedAt { get; set; } = string.Empty;
}