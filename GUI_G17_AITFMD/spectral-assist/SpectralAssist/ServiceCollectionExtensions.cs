using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Services;
using SpectralAssist.ViewModels;

namespace SpectralAssist;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // Services (singleton — reused across image reloads)
        collection.AddSingleton<ImageLoadingService>();
        collection.AddSingleton<InferenceService>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
    }
}