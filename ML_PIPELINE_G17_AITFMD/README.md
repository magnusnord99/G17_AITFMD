# ML Pipeline — HistologyHSI-GB 3D CNN

Maskinlæringspipeline for klassifisering av glioblastomvev ved hjelp av hyperspektrale histologibilder (HSI). Pipelinen trener 3D CNN-modeller på wavelet- eller PCA-reduserte spektrale kuber og eksporterer til ONNX for bruk i SpectralAssist-applikasjonen.

Beste resultat: **ResNet 3D-CNN + Wavelet16 (db2)** — F1=0.921, ROI-AUC=0.907 (5-fold kryssvalidering på pasientnivå).

**Krav:** Python 3.11+, PyTorch 2.x, pywavelets, onnx. GPU/MPS anbefalt for trening.

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
│   ├── checkpoints/                # Beste modell-checkpoints fra enkelt-treningskjøringer
│   ├── exports/                    # ONNX-eksporter klar for SpectralAssist
│   ├── plots/                      # Trenings- og evalueringsplott
│   ├── reports/                    # Evalueringsrapporter og metrikker
│   └── logs/                       # Trenings- og inferenslogger
├── scripts/
│   ├── run_train.py                # Start enkelt treningskjøring
│   ├── run_kfold.py                # K-fold kryssvalidering
│   ├── run_eval.py                 # Evaluering av trent modell (ROI-nivå)
│   ├── run_eval_patch.py           # Evaluering av trent modell (patch-nivå)
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
├── requirements.txt
└── run_inference.py                # Inferens-inngangspunkt (historisk, erstattet av C#)
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

> **Datasett:** HistologyHSI-GB (Ortega et al., 2024) — 469 annoterte hyperspektrale histologibilder fra 13 glioblastompasienter, 826 spektrale kanaler. Tilgjengelig fra [TCIA](https://www.cancerimagingarchive.net/) under CC BY 4.0.

## Trening

### Enkelt treningskjøring

Treningskonfigen (`configs/train.yaml` eller en under `configs/train/`) styrer datasett, modell, optimizer og treningsparametere.

```bash
# Standard enkelt kjøring (bruker configs/train.yaml)
python scripts/run_train.py --config configs/train.yaml

# Override modell-arkitektur uten å endre konfigen
python scripts/run_train.py --config configs/train/wavelet/baseline.yaml \
    --model configs/models/resnet_3dcnn.yaml

# Hopp over automatisk evaluering etter trening
python scripts/run_train.py --config configs/train.yaml --no-auto-eval
```

**Viktige felter i treningskonfigen:**
- `model_config` — sti til arkitektur-YAML (`configs/models/`)
- `data.cube_root` — rotmappe med reduserte `.npy`-kuber
- `data.cube_manifest_csv` — CSV med kubeindeks og klasselabels
- `data.patch_h / patch_w` — patchstørrelse (må matche modellens forventede input)
- `trainer.max_epochs` og `trainer.early_stopping_patience`

Checkpoints lagres til `outputs/checkpoints/<navn>_<dato>_best.pt`.

### K-fold kryssvalidering

K-fold-kjøringen trener én modell per fold og evaluerer automatisk. Anbefalt for endelig modellvalg.

```bash
# 5-fold kryssvalidering (standard)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml

# Kjør kun spesifikke folds (nyttig ved avbrutt kjøring)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml --folds 2,3,4

# Fortsett en avbrutt k-fold-kjøring (folds med fold_result.json hoppes over)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml \
    --resume outputs/training/kfold_resnet_3dcnn_20260419_210955
```

**Obligatoriske argumenter:** `--config`

Resultater per fold lagres under `outputs/training/<run_name>/fold_<N>/`.

## ONNX-eksport

Eksporterer et trent checkpoint til ONNX-format med tilhørende `manifest.json` for SpectralAssist.

### Obligatoriske argumenter

| Argument | Beskrivelse |
|----------|-------------|
| `--checkpoint` | Sti til `.pt`-checkpoint |
| `--out-dir` | Utdatamappe (opprettes hvis den ikke finnes) |
| `--spectral-bands` | Antall spektrale bånd **etter** reduksjon (f.eks. `16`). For PCA må dette være lik `n_components` i PCA-modellen. |
| `--patch-h` / `--patch-w` | Patchstørrelse i piksler — må matche det modellen ble trent på (vanligvis `64 64`) |

### Valgfrie argumenter

| Argument | Standard | Beskrivelse |
|----------|----------|-------------|
| `--reducer-method` | Fra `pipeline.yaml` | Overstyrer spektral reduksjonsmetode: `pca`, `wavelet`, `ae` eller `none` |
| `--raw-bands` | — | Antall rå bånd FØR reduksjon (f.eks. `275`). Nødvendig for wavelet/ae slik at `spectral_reducer.input_bands` i manifest blir riktig. For PCA leses dette automatisk fra PCA-modellen. |
| `--onnx-name` | `model.onnx` | Filnavn på ONNX-filen |
| `--model-config` | Fra checkpoint | Override modell-YAML hvis checkpointet mangler `model_config_path` |
| `--pipeline-config` | `configs/preprocessing/pipeline.yaml` | Preprosesseringskonfig som manifest leses fra |
| `--validation-roi-dir` | — | ROI-mappe med rå ENVI-filer — se avsnittet under |
| `--description` | — | Fritekstbeskrivelse i manifest |

### Valideringskube (`--validation-roi-dir`)

Valideringskuben er en rå HSI-ROI fra databasen (med `raw`, `darkReference` og `whiteReference` ENVI-filer). Eksport-scriptet:

1. Sliserer ut en patch fra midten av kuben (eller fra angitt `--validation-patch-y/x`)
2. Kjører Python-preprosesseringspipelinen på patchen (kalibrering → clipping → avg3 → spektral reduksjon)
3. Sender patchen gjennom PyTorch-modellen og lagrer de forventede logits/softmax

Dette skrives til `roi_validation/` i eksportmappen og legges inn i `manifest.json` under `validation.expected_output`. SpectralAssist bruker dette til å verifisere at C#-pipelinen gir identisk output som Python-pipelinen.

**Mappen må inneholde:**
```
<roi-dir>/
├── raw
├── raw.hdr
├── darkReference
├── darkReference.hdr
├── whiteReference
└── whiteReference.hdr
```

### Eksempler

```bash
# Wavelet-modell (vanligste tilfelle)
python scripts/export/export_cnn3d_onnx.py \
    --checkpoint outputs/checkpoints/resnet_3dcnn_20260402_best.pt \
    --out-dir outputs/exports/resnet_wavelet \
    --spectral-bands 16 \
    --patch-h 64 --patch-w 64 \
    --reducer-method wavelet \
    --raw-bands 275 \
    --validation-roi-dir "/Volumes/DJI/HSI_testing/npj_database/HSI_Human_Brain_Database_IEEE_Access/026-02"

# PCA-modell — PCA bakes inn i ONNX-grafen, ingen ekstern PCA-fil trengs i C#
python scripts/export/export_cnn3d_onnx.py \
    --checkpoint outputs/checkpoints/baseline_3dcnn_20260331_best.pt \
    --out-dir outputs/exports/baseline_pca \
    --spectral-bands 16 \
    --patch-h 64 --patch-w 64 \
    --reducer-method pca \
    --validation-roi-dir "/Volumes/DJI/HSI_testing/npj_database/HSI_Human_Brain_Database_IEEE_Access/026-02"
```

**Output:**
```
outputs/exports/<navn>/
├── model.onnx          # ONNX-modell (inkl. PCA-vekter hvis reducer=pca)
├── manifest.json       # Metadata, preprocessing-parametere, forventede logits
└── roi_validation/     # Utsnitt av valideringskuben (raw/dark/white ENVI)
    ├── raw + raw.hdr
    ├── darkReference + darkReference.hdr
    └── whiteReference + whiteReference.hdr
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

Rådata (ENVI hyperspektrale kuber) ligger i `PKG - HistologyHSI-GB/` og må ikke endres. Forprosessering genererer reduserte `.npy`-kuber (wavelet eller PCA) til `data/processed/` og mellomprodukter til `data/interim/`.

Forhåndsberegnede modeller for inferens (`models/`):
- `pca_avg3_16.joblib` — PCA-modell (16 komponenter, trenings-gjennomsnitt for whitening)
- `ae_avg3_16.pt` — Autoencoder (AVG-metode, 16 kanaler)

## SpectralAssist-integrasjon

Ferdigtrente modeller lastes inn av SpectralAssist (Avalonia C#-applikasjon i `GUI_G17_AITFMD/spectral-assist/`). En eksportmappe med `model.onnx`, `manifest.json` og `roi_validation/` utgjør en komplett modellpakke som kopieres inn i applikasjonen.

`manifest.json` inneholder bl.a.:
- `input_spec` — forventet tensorform og antall spektrale bånd
- `pipeline.preprocessing.steps` — hvilke preprosesseringssteg C# skal kjøre (f.eks. `calibrate`, `clip`, `neighbor_average`, `tissue_mask`)
- `pipeline.spectral_reducer` — reduksjonsmetode og om den er bakt inn i ONNX (`embedded_in_onnx: true` for PCA)
- `validation.expected_output` — forventede logits/softmax for referansepatchen, brukt til integrasjonstest i C#
