using System.Collections.Generic;
using System.Text.Json.Serialization;

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