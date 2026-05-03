using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;

namespace SpectralAssist.Extensions;

public static class ApplicationExtensions
{
    public static Window? MainWindow(this Application app) =>
        app.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime { MainWindow: { } window }
            ? window
            : null;
}