using Avalonia;
using Avalonia.Controls;

namespace SpectralAssist.Views.Components;

public partial class SidebarLayout : UserControl
{
    public static readonly StyledProperty<GridLength> SidebarWidthProperty =
        AvaloniaProperty.Register<SidebarLayout, GridLength>(
            nameof(SidebarWidth), new GridLength(260));

    public static readonly StyledProperty<object?> SidebarHeaderProperty =
        AvaloniaProperty.Register<SidebarLayout, object?>(nameof(SidebarHeader));

    public static readonly StyledProperty<object?> SidebarContentProperty =
        AvaloniaProperty.Register<SidebarLayout, object?>(nameof(SidebarContent));

    public static readonly StyledProperty<object?> SidebarActionsProperty =
        AvaloniaProperty.Register<SidebarLayout, object?>(nameof(SidebarActions));

    public static readonly StyledProperty<object?> MainContentProperty =
        AvaloniaProperty.Register<SidebarLayout, object?>(nameof(MainContent));

    public GridLength SidebarWidth
    {
        get => GetValue(SidebarWidthProperty);
        set => SetValue(SidebarWidthProperty, value);
    }

    public object? MainContent
    {
        get => GetValue(MainContentProperty);
        set => SetValue(MainContentProperty, value);
    }

    public object? SidebarHeader
    {
        get => GetValue(SidebarHeaderProperty);
        set => SetValue(SidebarHeaderProperty, value);
    }

    public object? SidebarContent
    {
        get => GetValue(SidebarContentProperty);
        set => SetValue(SidebarContentProperty, value);
    }

    public object? SidebarActions
    {
        get => GetValue(SidebarActionsProperty);
        set => SetValue(SidebarActionsProperty, value);
    }

    public SidebarLayout()
    {
        InitializeComponent();
    }
}
