using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.ViewModels;

namespace SpectralAssist;

public static class ServiceCollectionExtensions
{
    public static void AddCommonServices(this IServiceCollection collection)
    {
        // Services
        //collection.AddSingleton<IHsiLoader, HsiLoader>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
    }
}