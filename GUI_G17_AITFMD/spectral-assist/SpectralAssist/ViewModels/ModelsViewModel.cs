using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using SpectralAssist.Models;
using SpectralAssist.Services;

namespace SpectralAssist.ViewModels;

/// <summary>The three states of the Models page.</summary>
public enum ModelViewState { Browsing, Previewing, Importing, }

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
    private readonly ModelPackageService _modelRegistry;
    private readonly InferenceService _inferenceService;
    private readonly Action<ModelManifest?> _setActiveModel;
    public ObservableCollection<ModelManifest> AvailableModels => _modelRegistry.AvailableModels;
    private string? _importSourcePath;

    public ModelsViewModel(ModelPackageService modelRegistry, InferenceService inferenceService,
        ModelManifest? activeModel, Action<ModelManifest?> setActiveModel)
    {
        _modelRegistry = modelRegistry;
        _inferenceService = inferenceService;
        _setActiveModel = setActiveModel;
        SelectedModel = activeModel ?? AvailableModels.FirstOrDefault();
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
    private ModelManifest? _selectedModel;

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(DisplayedModel))]
    private ModelManifest? _previewModel;

    [ObservableProperty] private string? _errorMessage;
    [ObservableProperty] private string? _successMessage;

    public ModelManifest? DisplayedModel =>
        ViewState is ModelViewState.Previewing or ModelViewState.Importing
            ? PreviewModel
            : SelectedModel;


    // -- Actions -- //

    [RelayCommand]
    private void DismissError() => ErrorMessage = null;
    
    [RelayCommand]
    private void DismissSuccess() => ErrorMessage = null;

    public void PreviewImport(string folderPath)
    {
        ErrorMessage = null;
        SuccessMessage = null;
        
        var result = ModelPackageService.TryLoadManifest(folderPath);
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

        var importResult = await Task.Run(() => _modelRegistry.ImportPackage(_importSourcePath));
        if (!importResult.IsSuccess)
        {
            ErrorMessage = importResult.Error;
            ResetImportState();
            return;
        }
        
        var modelManifest = importResult.Value!;
        var (passed, summary) = await ModelPackageValidator.ValidateAsync(
            modelManifest, _modelRegistry, _inferenceService);

        if (passed)
            SuccessMessage = summary;
        else
            ErrorMessage = summary;

        ResetImportState();
        SelectedModel = modelManifest;
    }

    [RelayCommand]
    private void CancelImport() => ResetImportState();

    [RelayCommand]
    private void DeleteModel(ModelManifest modelInfo)
    {
        var result = _modelRegistry.DeletePackage(modelInfo.Id);
        if (!result.IsSuccess)
        {
            ErrorMessage = result.Error;
            return;
        }

        if (SelectedModel == modelInfo)
            SelectedModel = AvailableModels.FirstOrDefault(); 
        //ToDo: SelectedModel = AvailableModels.Count > 0 ? AvailableModels[0] : null;
    }
    
    partial void OnSelectedModelChanged(ModelManifest? value)
    {
        if (IsBrowsing)
            _setActiveModel(value);
    }
    
    private void ResetImportState()
    {
        PreviewModel = null;
        _importSourcePath = null;
        ViewState = ModelViewState.Browsing;
    }


    /// <summary>Design preview constructor filled with dummy data.</summary>
    public ModelsViewModel()
    {
        _modelRegistry = new ModelPackageService();
        _setActiveModel = _ => { };
        var sample = new ModelManifest
        {
            Id = "baseline_3dcnn_20260324",
            DirectoryPath = "",
            Metadata = new ManifestMetadata
            {
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

        _modelRegistry.AvailableModels.Add(sample);
        _selectedModel = sample;
    }
}