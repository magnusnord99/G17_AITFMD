using System.Collections.Generic;
using System.Text.Json.Serialization;
using SpectralAssist.Services.Preprocessing;

namespace SpectralAssist.Models;

public class ModelManifest
{
    [JsonPropertyName("schema_version")]
    public string SchemaVersion { get; set; } = "";

    [JsonPropertyName("generator")]
    public string Generator { get; set; } = "";

    [JsonPropertyName("metadata")]
    public ModelPackageMetadata Metadata { get; set; } = new();

    [JsonPropertyName("training_config")]
    public TrainingConfig TrainingConfig { get; set; } = new();

    [JsonPropertyName("input_spec")]
    public InputSpec InputSpec { get; set; } = new();

    [JsonPropertyName("output_spec")]
    public OutputSpec OutputSpec { get; set; } = new();

    [JsonPropertyName("artifacts")]
    public ArtifactPaths Artifacts { get; set; } = new();

    [JsonPropertyName("preprocessing_config")]
    public PreprocessingConfig? PreprocessingConfig { get; set; }

    [JsonPropertyName("validation")]
    public ValidationSpec? Validation { get; set; }
}

public class ModelPackageMetadata
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("version")]
    public string Version { get; set; } = "";

    [JsonPropertyName("created")]
    public string Created { get; set; } = "";

    [JsonPropertyName("author")]
    public string Author { get; set; } = "";

    [JsonPropertyName("description")]
    public string Description { get; set; } = "";
}

public class TrainingConfig
{
    [JsonPropertyName("preprocessing")]
    public string Preprocessing { get; set; } = "";

    [JsonPropertyName("model_type")]
    public string ModelType { get; set; } = "";

    [JsonPropertyName("dataset")]
    public string Dataset { get; set; } = "";

    [JsonPropertyName("samples")]
    public int Samples { get; set; }

    [JsonPropertyName("epochs")]
    public int Epochs { get; set; }

    [JsonPropertyName("val_accuracy")]
    public double ValAccuracy { get; set; }

    [JsonPropertyName("classes")]
    public List<string> Classes { get; set; } = [];
}

public class InputSpec
{
    /// <summary>4 = NCHW (eldre 2D-flyt); 5 = NCDHW for 3D-CNN (1,1,C,H,W).</summary>
    [JsonPropertyName("input_rank")]
    public int? InputRank { get; set; }

    [JsonPropertyName("tensor_layout")]
    public string TensorLayout { get; set; } = "";

    /// <summary>Eks.: [1, 1, 16, 64, 64] for statisk 3D-CNN-eksport.</summary>
    [JsonPropertyName("input_shape")]
    public List<int>? InputShape { get; set; }

    /// <summary>Alias for spektral dybde ved 3D-CNN (kan være lik <see cref="ExpectedBands"/>).</summary>
    [JsonPropertyName("spectral_bands")]
    public int? SpectralBands { get; set; }

    [JsonPropertyName("expected_bands")]
    public int ExpectedBands { get; set; }

    [JsonPropertyName("wavelength_range_nm")]
    public List<double> WavelengthRangeNm { get; set; } = [];

    [JsonPropertyName("spatial_patch_size")]
    public List<int> SpatialPatchSize { get; set; } = [];

    [JsonPropertyName("dtype")]
    public string Dtype { get; set; } = "float32";
}

public class OutputSpec
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    [JsonPropertyName("classes")]
    public List<string> Classes { get; set; } = [];
}

public class ArtifactPaths
{
    [JsonPropertyName("pipeline_onnx")]
    public string PipelineOnnx { get; set; } = "";
}

public class PreprocessingConfig
{
    [JsonPropertyName("calibration_epsilon")]
    public float CalibrationEpsilon { get; set; }

    [JsonPropertyName("clip_min")]
    public float ClipMin { get; set; }

    [JsonPropertyName("clip_max")]
    public float ClipMax { get; set; } = 1f;

    [JsonPropertyName("neighbor_average_window")]
    public int NeighborAverageWindow { get; set; } = 3;

    [JsonPropertyName("band_reduce_out_bands")]
    public int BandReduceOutBands { get; set; } = 16;

    [JsonPropertyName("band_reduce_strategy")]
    public string BandReduceStrategy { get; set; } = "crop";

    [JsonPropertyName("tissue_mask_method")]
    public string TissueMaskMethod { get; set; } = "mean_std_percentile";

    [JsonPropertyName("tissue_mask_q_mean")]
    public float TissueMaskQMean { get; set; } = 0.5f;

    [JsonPropertyName("tissue_mask_q_std")]
    public float TissueMaskQStd { get; set; } = 0.4f;

    [JsonPropertyName("tissue_mask_min_object_size")]
    public int TissueMaskMinObjectSize { get; set; } = 1000;

    [JsonPropertyName("tissue_mask_min_hole_size")]
    public int TissueMaskMinHoleSize { get; set; } = 1000;

    /// <summary>
    /// Ordered list of preprocessing steps to execute.
    /// If null/empty, the runner uses default steps:
    /// ["calibrate", "clip", "neighbor_average", "tissue_mask", "band_average"].
    /// </summary>
    [JsonPropertyName("steps")]
    public List<string>? Steps { get; set; }

    /// <summary>Converts tissue mask parameters to a <see cref="TissueMaskOptions"/>.</summary>
    public TissueMaskOptions ToTissueOptions() => new(
        qMean: TissueMaskQMean,
        qStd: TissueMaskQStd,
        minObjectSize: TissueMaskMinObjectSize,
        minHoleSize: TissueMaskMinHoleSize);
}

public class ValidationSpec
{
    [JsonPropertyName("ref_cube_shape")]
    public List<int> RefCubeShape { get; set; } = [];

    [JsonPropertyName("expected_output_shape")]
    public List<int> ExpectedOutputShape { get; set; } = [];

    [JsonPropertyName("preprocessing_tolerance")]
    public float PreprocessingTolerance { get; set; } = 1e-5f;

    [JsonPropertyName("inference_tolerance")]
    public float InferenceTolerance { get; set; } = 1e-4f;

    [JsonPropertyName("layout")]
    public string Layout { get; set; } = "hwb_c_order";

    [JsonPropertyName("dtype")]
    public string Dtype { get; set; } = "float32";

    [JsonPropertyName("num_patches")]
    public int NumPatches { get; set; } = 1;

    [JsonPropertyName("patch_probs_shape")]
    public List<int> PatchProbsShape { get; set; } = [];
}