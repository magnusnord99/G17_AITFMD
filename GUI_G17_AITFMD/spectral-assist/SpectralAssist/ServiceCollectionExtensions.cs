using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Services;
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

        // ViewModels
        collection.AddSingleton<MainViewModel>();
    }
}