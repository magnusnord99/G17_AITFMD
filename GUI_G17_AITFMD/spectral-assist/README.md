# SpectralAssist
![.NET](https://img.shields.io/badge/.NET-10-512bd4?logo=dotnet)
![Avalonia](https://img.shields.io/badge/AvaloniaUI-Desktop-purple?logo=dotnet)
![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Win-success)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

**A cross-platform desktop application for medical anomaly detection in hyperspectral imaging (HSI).**

SpectralAssist is built with **Avalonia UI** and **.NET 10 (LTS)**. It provides a high-performance graphical user interface for visualizing HSI cubes and detecting medical anomalies using machine-learning models. 

The application serves as the primary inference and visualization frontend for the [ML_Pipeline_G17](https://github.com/magnusnord99/ML_Pipeline_G17_AITFMD) engine, bridging the gap between machine-learning research and clinical application.

(NYI) To enable inference, place your exported model files from ML_Pipeline_G17 in `/Models/` directory or import them using the Import Model function inside the GUI.
The application will automatically detect all available models at startup.

## Getting Started for Developers
1. Install .NET 10 (LTS).
2. Clone the Repository.
3. Run Application:
   - CLI: Navigate to Project Folder and Run: `dotnet run --project SpectralAssist`
   - IDE: Open `SpectralAssist.sln` and hit Run

## End-User Installation
**Prebuilt binaries will be provided for:**
- Linux: SpectralAssist-linux.tar.gz
- macOS: SpectralAssist-macos.zip
- Windows: SpectralAssist-win64.zip

**Extract the downloaded archive and run:**
- macOS / Linux: ./SpectralAssist
- Windows: SpectralAssist.exe

No additional runtime dependencies are required.
