using Avalonia;
using Avalonia.Controls;

namespace SpectralAssist.Views.Components;

public partial class LabeledRow : UserControl
{
    public static readonly StyledProperty<string?> LabelProperty =
        AvaloniaProperty.Register<LabeledRow, string?>(nameof(Label));

    public static readonly StyledProperty<string?> ValueProperty =
        AvaloniaProperty.Register<LabeledRow, string?>(nameof(Value));

    public string? Label
    {
        get => GetValue(LabelProperty);
        set => SetValue(LabelProperty, value);
    }

    public string? Value
    {
        get => GetValue(ValueProperty);
        set => SetValue(ValueProperty, value);
    }

    public LabeledRow()
    {
        InitializeComponent();
    }
}
