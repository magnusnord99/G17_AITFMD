using System;
using Microsoft.Extensions.DependencyInjection;
using SpectralAssist.Models;
using SpectralAssist.Services;
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

        collection.AddSingleton<IDialogService, DialogService>();

        // ViewModels
        collection.AddSingleton<MainViewModel>();
        collection.AddSingleton<LibraryViewModel>();
        collection.AddSingleton<ModelsViewModel>();

        // ImageViewModel Factory
        collection.AddSingleton<Func<ImageNode, ImageViewModel>>(sp => (imageNode) =>
            new ImageViewModel(
                imageNode,
                sp.GetRequiredService<InferenceService>(),
                sp.GetRequiredService<LibraryManager>(),
                sp.GetRequiredService<SessionService>(),
                sp.GetRequiredService<IDialogService>()));
    }
}