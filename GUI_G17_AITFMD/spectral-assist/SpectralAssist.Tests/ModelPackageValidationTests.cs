using SpectralAssist.Services;
using SpectralAssist.Services.Inference;
using Xunit;

namespace SpectralAssist.Tests;

/// <summary>
/// Build-time test that runs the same import-time validation the runtime uses when user imports a model package.
/// Failures here mean preprocessing, inference, or the validator itself has drifted
/// meaning real model imports will also report a mismatch.
/// </summary>
public class ModelPackageValidationTests
{
    private const string ReferencePackageName = "msd_dense_v3";
    private static string LocateReferencePackage() =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", ReferencePackageName);

    [Fact]
    public async Task Reference_package_passes_end_to_end_validation()
    {
        var packageDir = LocateReferencePackage();
        Assert.True(Directory.Exists(packageDir),
            $"Reference package not found at '{packageDir}'. " +
            "Ensure SpectralAssist/ModelPackages/" + ReferencePackageName + " is checked in.");

        var manifestResult = ModelPackageManager.TryLoadManifest(packageDir);
        Assert.True(manifestResult.IsSuccess, manifestResult.Error);

        var packageService = new ModelPackageManager();
        var session = new SessionService();
        var classifier = new Onnx3DCnnClassifier();
        var inferenceService = new InferenceService(classifier, session, packageService);

        var (passed, summary) = await ModelPackageValidator.ValidateAsync(
            manifestResult.Value!, packageService, inferenceService);

        Assert.True(passed, summary);
    }
}
