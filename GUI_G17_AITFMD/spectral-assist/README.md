# SpectralAssist

Cross-platform desktop application for visualizing and analyzing
hyperspectral histology images, with ONNX-based anomaly detection.

The app consumes model packages; each a directory containing an
ONNX model, a `manifest.json` describing the preprocessing pipeline,
and an optional validation folder with a small HSI patch used for
an import-time smoke test. Packages are produced by the Python ML
pipeline in the sibling project in this monorepo. Several default
packages ship with the application, so importing your own is
optional.

## Quick Start

### Requirements

- .NET 10 SDK
- One or more HSI images (`.hdr` + raw cube files) to open or build
  a library from. The application ships without sample data.
- (Optional) NVIDIA CUDA 12.9 + cuDNN 9.10.2 for GPU-accelerated inference

### Running from Source

Open `SpectralAssist.sln` in Visual Studio or JetBrains Rider and
run the SpectralAssist project, or from the command line:

```bash
dotnet run --project SpectralAssist
```

## Building Distribution Packages

### Option 1: Avalonia Parcel (recommended for releases)

Open `SpectralAssist/SpectralAssist.parcel` in the
[Avalonia Parcel GUI](https://docs.avaloniaui.net/tools/parcel/setup)
and run the build. Platform targets, icons, installer metadata,
and single-file packaging are pre-configured in the `.parcel` file.

Output packages are written to `SpectralAssist/bin/packages/`:

- Windows: `.exe` installer and `.zip` archive
- macOS: `.dmg` image and `.zip` archive
- Linux: `.deb`, `.rpm` and `.zip` archive

> **Note:** Avalonia Parcel may require a free Avalonia account to
> activate. If you only need a runnable binary without installer
> packaging, use Option 2 instead.

### Option 2: Manual `dotnet publish` (no account required)

Produces a standalone executable per platform, without the
installer and icon polish from Parcel:

```bash
# Windows x64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r win-x64 -p:PublishSingleFile=true -p:SelfContained=true

# macOS arm64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r osx-arm64 -p:PublishSingleFile=true -p:SelfContained=true

# Linux x64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r linux-x64 -p:PublishSingleFile=true -p:SelfContained=true
```
Output is written to `SpectralAssist/bin/Release/net10.0/<rid>/publish/`.

## Project Structure

```
SpectralAssist/
├── SpectralAssist/              Main application project
│   ├── Assets/                  Icons and image resources
│   ├── Extensions/              C# extension methods
│   ├── Models/                  Domain data models
│   ├── ModelPackages/           Default ONNX model packages
│   ├── Services/                Domain services (loading, preprocessing,
│   │                            inference, library, export, rendering)
│   ├── Styles/                  Avalonia XAML styles
│   ├── ViewModels/              MVVM view models
│   ├── Views/                   Avalonia XAML views
│   └── SpectralAssist.parcel    Avalonia Parcel configuration
├── SpectralAssist.Tests/        Unit tests
├── LICENSE                      Apache License 2.0
└── README.md                    This file
```

## Model Packages

Default model packages are bundled under
`SpectralAssist/ModelPackages/` and loaded automatically on startup.
Additional packages can be imported at runtime from the Models view
in the application.

Each package is a directory laid out like the bundled
`resnet_3dcnn` example:

```
resnet_3dcnn/
├── resnet_3dcnn_final.onnx       The ONNX model graph
├── resnet_3dcnn_final.onnx.data  External weights
├── manifest.json                 Pipeline, input/output specs,
│                                 training metadata, evaluation metrics
└── roi_validation/               Optional smoke-test patch
    ├── raw.hdr + raw             HSI scene
    ├── darkReference.hdr + …     Dark calibration reference
    └── whiteReference.hdr + …    White calibration reference
```

The `roi_validation/` folder is run through the full
preprocess and inference pipeline at import time and its output compared
against expected values stored in `manifest.json`. Validation is
warn-only; a failed check does not block use of the package.

## Documentation

For end-user documentation, installation instructions, and a
walkthrough of the application's features, see the
**Systemdokumentasjon** appendix in the bachelor thesis included
with this codebase.
