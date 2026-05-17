# ML Pipeline — HistologyHSI-GB 3D CNN

Maskinlæringspipeline for klassifisering av glioblastomvev ved hjelp av hyperspektrale histologibilder (HSI). Pipelinen trener 3D CNN-modeller på PCA-reduserte spektrale kuber og eksporterer modeller til ONNX for bruk i SpectralAssist-applikasjonen.

## Prosjektstruktur

```
ML_PIPELINE_G17_AITFMD/
├── configs/
│   ├── train.yaml                  # Hoved-treningskonfig
│   ├── models/                     # Arkitekturkonfiger (baseline, resnet, msd_dense, ad_hybrid_sn)
│   ├── preprocessing/              # PCA- og pipeline-konfiger
│   ├── grid_search/                # Grid search-konfiger
│   └── inference/                  # Inferenskonfiger
├── data/
│   ├── raw/                        # Rådata (ENVI-kuber fra PKG-mappen)
│   ├── interim/                    # Mellomprodukt: masker, manifester, PCA-modeller
│   ├── processed/                  # PCA-reduserte kuber klar for trening
│   └── splits/                     # Tog/val/test-splitter (JSON)
├── models/
│   ├── ae_avg3_16.pt               # Autoencoder for inferens
│   └── pca_avg3_16.joblib          # PCA-modell for inferens
├── outputs/
│   ├── training/                   # Alle treningskjøringer (k-fold per modell)
│   ├── onnx/
│   │   ├── final/                  # Produksjonsklare ONNX-modeller
│   │   ├── kfold/                  # K-fold eksporterte ONNX-modeller
│   │   ├── export/                 # Eksport-tester
│   │   └── experimental/          # Eksperimentelle eksporter
│   ├── checkpoints/                # Beste modell-checkpoints
│   ├── plots/                      # Trenings- og evalueringsplott
│   ├── reports/                    # Evalueringsrapporter og metrikker
│   ├── logs/                       # Trenings- og inferenslogger
│   └── test/                       # Testutdata og grid search-resultater
├── scripts/
│   ├── run_train.py                # Start enkelt treningskjøring
│   ├── run_kfold.py                # K-fold kryssvalidering
│   ├── run_eval.py                 # Evaluering av trent modell
│   ├── preprocessing/              # Forprosesseringsskript (PCA, masker, indeksering)
│   └── export/                     # ONNX-eksportskript
├── src/
│   ├── datasets/                   # Dataset-klasser (CubePatchDataset)
│   ├── models/cnn3d/               # Modellarkitekturer (baseline, resnet, msd_dense, ad_hybrid_sn)
│   ├── preprocessing/              # PCA, autoencoder, vevsmaskering, patching
│   ├── training/                   # Treningsloop, callbacks, early stopping
│   ├── evaluation/                 # Evalueringspipeline, metrikker, rapportering
│   ├── inference/                  # Inferenspipeline og heatmap-generering
│   └── utils/                      # Logging og hjelpefunksjoner
├── tests/                          # Enhetstester
├── docs/visualizations/            # Visualiseringsskript
├── requirements.txt
└── run_inference.py                # Inferens-inngangspunkt
```

## Oppsett

```bash
cd ML_PIPELINE_G17_AITFMD
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kontroller at stier i `configs/preprocessing/pipeline.yaml` peker til riktig datasettlokasjon (`PKG - HistologyHSI-GB`).

## Vanlige kommandoer

```bash
# Forprosessering (PCA-reduksjon + vevsmaskering)
python scripts/preprocessing/run_pipeline.py --config configs/preprocessing/pipeline.yaml

# Trening (enkelt kjøring)
python scripts/run_train.py --config configs/train.yaml

# K-fold kryssvalidering
python scripts/run_kfold.py --config configs/train.yaml

# Evaluering
python scripts/run_eval.py --config configs/train.yaml

# ONNX-eksport
python scripts/export/export_cnn3d_onnx.py

# Inferens på ny ROI
python run_inference.py --input <sti-til-kube> --output-dir outputs/
```

## Modeller

Fire 3D CNN-arkitekturer er implementert og evaluert:

| Modell | Beskrivelse |
|--------|-------------|
| `baseline_3dcnn` | Enkel 3D CNN med konvolusjonsblokker |
| `resnet_3dcnn` | ResNet-inspirert med residualforbindelser |
| `msd_dense_3dcnn` | Multi-scale dense connections |
| `ad_hybrid_sn` | Hybrid spektral-romlig arkitektur |

Modellkonfiger velges via `model_config` i `configs/train.yaml`.

## Data

Rådata (ENVI hyperspektrale kuber) ligger i `PKG - HistologyHSI-GB/` og må ikke endres. Forprosessering genererer PCA-reduserte `.npy`-kuber til `data/processed/` og mellomprodukter til `data/interim/`.

## ONNX-eksport og SpectralAssist

Ferdigtrente modeller eksporteres til ONNX-format og lastes inn av SpectralAssist (C# WPF-applikasjon i `GUI_G17_AITFMD/`). Produksjonsklare modeller ligger i `outputs/onnx/final/`.
