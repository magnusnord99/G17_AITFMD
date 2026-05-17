using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

/// <summary>The three states of the Models page.</summary>
public enum ModelViewState
{ Browsing, Previewing, Importing }

/// <summary>
/// Controls the three states of the Models page:
/// <list>
/// <item><b>Browsing</b>: Dropdown of imported models, detail view, delete button.</item>
/// <item><b>Previewing</b>: User picked a folder, manifest shown for review, confirm/cancel.</item>
/// <item><b>Importing</b>: Files being copied + validation running, progress shown.</item>
/// </list>
/// </summary>
public partial class ModelsViewModel : ViewModelBase
{
    private readonly ModelPackageManager _modelManager;
    private readonly InferenceService _inferenceService;
    private string? _importSourcePath;

    public ObservableCollection<ModelManifest> AvailableModels => _modelManager.AvailableModels;
    public SessionService Session { get; } 

    public ModelsViewModel(
        ModelPackageManager modelManager,
        InferenceService inferenceService,
        SessionService session)
    {
        _modelManager = modelManager;
        _inferenceService = inferenceService;
        Session = session;

        Session.ActiveModel ??= AvailableModels.FirstOrDefault();
        Session.PropertyChanged += OnSessionChanged;
    }

    // -- States -- //
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsBrowsing))]
    [NotifyPropertyChangedFor(nameof(IsPreviewing))]
    [NotifyPropertyChangedFor(nameof(IsImporting))]
    [NotifyPropertyChangedFor(nameof(DisplayedModel))]
    private ModelViewState _viewState = ModelViewState.Browsing;

    public bool IsBrowsing => ViewState == ModelViewState.Browsing;
    public bool IsPreviewing => ViewState == ModelViewState.Previewing;
    public bool IsImporting => ViewState == ModelViewState.Importing;
    
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(DisplayedModel))]
    private ModelManifest? _previewModel;

    [ObservableProperty] private string? _errorMessage;
    [ObservableProperty] private string? _successMessage;

    public ModelManifest? DisplayedModel =>
        ViewState is ModelViewState.Previewing or ModelViewState.Importing
            ? PreviewModel
            : Session.ActiveModel;
    
    // -- Actions -- //
    [RelayCommand]
    private void DismissError() => ErrorMessage = null;

    [RelayCommand]
    private void DismissSuccess() => SuccessMessage = null;

    public void PreviewImport(string folderPath)
    {
        ErrorMessage = null;
        SuccessMessage = null;

        var result = ModelPackageManager.TryLoadManifest(folderPath);
        if (!result.IsSuccess)
        {
            ErrorMessage = result.Error;
            return;
        }

        _importSourcePath = folderPath;
        PreviewModel = result.Value;
        ViewState = ModelViewState.Previewing;
    }

    [RelayCommand]
    private async Task ConfirmImport()
    {
        if (_importSourcePath == null) return;

        ViewState = ModelViewState.Importing;
        ErrorMessage = null;
        SuccessMessage = null;

        var importResult = await Task.Run(() => _modelManager.ImportPackage(_importSourcePath));
        if (!importResult.IsSuccess)
        {
            ErrorMessage = importResult.Error;
            ResetImportState();
            return;
        }

        var modelManifest = importResult.Value!;
        var (passed, summary) = await ModelPackageValidator.ValidateAsync(
            modelManifest, _modelManager, _inferenceService);

        if (passed)
            SuccessMessage = summary;
        else
            ErrorMessage = summary;

        ResetImportState();
        Session.ActiveModel = modelManifest;
    }

    [RelayCommand]
    private void CancelImport() => ResetImportState();

    [RelayCommand]
    private void DeleteModel(ModelManifest modelInfo)
    {
        var result = _modelManager.DeletePackage(modelInfo.Metadata.Id);
        if (!result.IsSuccess)
        {
            ErrorMessage = result.Error;
            return;
        }
        
        if (Session.ActiveModel?.Metadata.Id == modelInfo.Metadata.Id)
            Session.ActiveModel = AvailableModels.FirstOrDefault();
    }
    
    private void ResetImportState()
    {
        PreviewModel = null;
        _importSourcePath = null;
        ViewState = ModelViewState.Browsing;
    }
    
    private void OnSessionChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(SessionService.ActiveModel))
            OnPropertyChanged(nameof(DisplayedModel));
    }
    
    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ModelsViewModel()
    {
        _modelManager = new ModelPackageManager();
        _inferenceService = null!;
        Session = new SessionService();

        var sample = new ModelManifest
        {
            DirectoryPath = "",
            Metadata = new ManifestMetadata
            {
                Id = "baseline_3dcnn_20260324",
                Name = "Baseline 3D-CNN",
                Version = "1.0.0",
                Description = "3D convolutional classifier for hyperspectral tissue analysis.",
                Author = "G17 AITFMD",
                Created = "2026-03-24",
            },
            Pipeline = new PipelineInfo
            {
                SpectralReducer = new SpectralReducerInfo { Method = "band_average" },
                Model = new ModelInfo
                {
                    Architecture = "baseline_3dcnn",
                    Task = "classification",
                    TotalParameters = 284_930,
                },
            },
            InputSpec = new InputSpec
            {
                SpectralBands = 16,
                SpatialPatchSize = [32, 32],
            },
            OutputSpec = new OutputSpec
            {
                Classes = ["normal", "anomaly"]
            },
            Training = new TrainingInfo
            {
                Dataset = "hsi_dataset_v1",
                Metrics = new TrainingMetrics { Accuracy = 0.938 },
            },
        };

        _modelManager.AvailableModels.Add(sample);
        Session.ActiveModel = sample;
    }
}