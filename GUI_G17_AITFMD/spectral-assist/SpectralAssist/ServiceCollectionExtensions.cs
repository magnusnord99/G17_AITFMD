using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Services;
using SpectralAssist.Services.Export;
using SpectralAssist.ViewModels;

namespace SpectralAssist;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // Services
        collection.AddSingleton<ImageLoadingService>();
        collection.AddSingleton<InferenceService>();
        collection.AddSingleton<ModelPackageService>();
        collection.AddSingleton<PdfReportService>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
    }
}