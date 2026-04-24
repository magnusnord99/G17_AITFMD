using System;
using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Services;
using SpectralAssist.Services.Export;
using SpectralAssist.Services.Inference;
using SpectralAssist.Services.Library;
using SpectralAssist.ViewModels;

namespace SpectralAssist;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // Services
        collection.AddSingleton<SessionService>();
        collection.AddSingleton<ImageLoadingService>();

        collection.AddSingleton<InferenceService>();
        collection.AddSingleton<Onnx3DCnnClassifier>();

        collection.AddSingleton<ModelPackageManager>();
        collection.AddSingleton<LibraryManager>();
        collection.AddSingleton<PdfReportService>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
        collection.AddSingleton<LibraryViewModel>();
        collection.AddSingleton<ModelsViewModel>();

        // ImageViewModel Factory
        collection.AddSingleton<Func<string, string?, ImageViewModel>>(sp => (path, imageId) =>
            new ImageViewModel(path, sp.GetRequiredService<InferenceService>(), imageId,
                sp.GetRequiredService<LibraryManager>(),
                sp.GetRequiredService<PdfReportService>()));
    }
}
