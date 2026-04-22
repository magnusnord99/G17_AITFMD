using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Services;
using SpectralAssist.Services.Library;
using SpectralAssist.ViewModels;

namespace SpectralAssist;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // Services
        collection.AddSingleton<ImageLoadingService>();
        collection.AddSingleton<InferenceService>();
        collection.AddSingleton<ModelPackageManager>();
        collection.AddSingleton<LibraryManager>();
        collection.AddSingleton<LibraryScanner>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
    }
}