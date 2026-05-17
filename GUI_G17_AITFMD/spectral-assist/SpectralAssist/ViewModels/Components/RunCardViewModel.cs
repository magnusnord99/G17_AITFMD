using System;
using CommunityToolkit.Mvvm.ComponentModel;
using SpectralAssist.Models;

namespace SpectralAssist.ViewModels.Components;

/// <summary>
/// View-wrapper around a <see cref="RunSummary"/> for the runs-list in ImageView.
/// Carries view-only state (currently just <see cref="IsActive"/>) so the XAML
/// can bind <c>Classes.active</c> and pill-visibility to a simple bool, instead
/// of needing parent-context look-ups or MultiBinding converters.
/// </summary>
public partial class RunCardViewModel(RunSummary run, bool isActive = false) : ObservableObject
{
    public RunSummary Run { get; } = run;

    [ObservableProperty] private bool _isActive = isActive;

    // Display properties
    public string RunId => Run.RunId;
    public string ModelDisplayName => Run.ModelDisplayName;
    public DateTime CompletedAt => Run.CompletedAt;
    public string PositiveClassName => Run.PositiveClassName;
    public double PositiveClassPercentAbove50 => Run.PositiveClassPercentAbove50;
    public double PositiveClassPercentAbove80 => Run.PositiveClassPercentAbove80;
}
