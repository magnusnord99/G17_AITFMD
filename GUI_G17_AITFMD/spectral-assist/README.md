# SpectralAssist

Cross-platform desktop application for visualizing and analyzing
hyperspectral histology images, with support for ONNX-based
anomaly detection models.

## Quick Start

### Requirements

- .NET 10 SDK
- (Optional) NVIDIA CUDA 12.9 + cuDNN 9.10 for GPU-accelerated inference

### Running from Source

```bash
git clone <repository-url>
cd SpectralAssist
dotnet run --project SpectralAssist
```

Or open `SpectralAssist.sln` in Visual Studio or JetBrains Rider
and run the SpectralAssist project.

## Building Distribution Packages

### Option 1: Avalonia Parcel (Recommended)

The project includes an Avalonia Parcel configuration that
automates building and packaging for all supported platforms
and formats. From the project root:

```bash
parcel build
```

This produces installation packages in `/dist/`:
- Windows: `.exe` installer and `.zip` archive
- macOS: `.dmg` image and `.zip` archive
- Linux: `.deb`, `.rpm` and `.zip` archive

The Parcel configuration file is located at `[FILBANE]/parcel.json`
and can be modified to adjust platform targets, package metadata,
or build options.

### Option 2: Manual `dotnet publish`

For more control over the build process, distribution binaries
can be built directly with `dotnet publish`:

```bash
# Windows x64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r win-x64 -o ./dist/win-x64

# macOS arm64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r osx-arm64 -o ./dist/macos-arm64

# Linux x64
dotnet publish SpectralAssist/SpectralAssist.csproj -c Release -r linux-x64 -o ./dist/linux-x64
```

Build flags such as `PublishSingleFile`, `SelfContained`, and
`SatelliteResourceLanguages` are pre-configured in
`SpectralAssist.csproj`.

## Project Structure

```
SpectralAssist/
├── SpectralAssist/              Main application project
│   ├── Views/                   Avalonia XAML views
│   ├── ViewModels/              MVVM view models
│   ├── Services/                Domain services (HSI, inference, etc.)
│   ├── Models/                  Domain data models
│   └── ModelPackages/           Default ONNX model packages
├── SpectralAssist.Tests/        Unit tests
├── parcel.json                  Avalonia Parcel configuration
└── README.md                    This file
```

## Documentation

For end-user documentation, installation instructions, and a
walkthrough of the application's features, see the
**Systemdokumentasjon** appendix in the bachelor thesis
included with this codebase.
