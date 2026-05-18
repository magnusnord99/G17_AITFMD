# ML Pipeline — HistologyHSI-GB 3D CNN

Maskinlæringspipeline for klassifisering av glioblastomvev ved hjelp av hyperspektrale histologibilder (HSI). Pipelinen trener 3D CNN-modeller på wavelet- eller PCA-reduserte spektrale kuber og eksporterer til ONNX for bruk i SpectralAssist-applikasjonen.

**Beste resultat: ResNet 3D-CNN + Wavelet16 (db2) — F1=0.921, ROI-AUC=0.907 (5-fold kryssvalidering på pasientnivå).**

**Krav:** Python 3.11+, PyTorch 2.x, pywavelets, onnx. GPU/MPS anbefalt for trening.

---

## Bakgrunn og problemstilling

Glioblastom (GBM) er den mest aggressive formen for hjernekreft. Ved operasjon er det kritisk å skille tumorbåndet fra friskt hjernevev i sanntid. Tradisjonell histologi krever farging og mikroskopi, noe som tar tid. Hyperspektrale kameraer kan derimot fange opp spektrale signaturer direkte fra vev — uten farging — og potensielt gi nevrokirurger rask veiledning under inngrepet.

Dette prosjektet bygger en ende-til-ende-pipeline som:

1. **Forprosesserer** rådata fra hyperspektrale ENVI-kuber (kalibrering, spektral reduksjon, vevsmaskering, patching)
2. **Trener** 3D CNN-modeller til å klassifisere 64×64-patcher som tumor eller frikt vev
3. **Evaluerer** modeller på ROI-nivå med kryssvalidering per pasient
4. **Eksporterer** til ONNX med tilhørende metadata slik at C#-applikasjonen (SpectralAssist) kan kjøre inferens

---

## Datasett

**HistologyHSI-GB** (Ortega et al., 2024) — 469 annoterte hyperspektrale histologibilder fra 13 glioblastompasienter.

- **826 spektrale kanaler** (bølgelengder) per piksel
- Rådata er ENVI-kuber med tilhørende mørk- og lysmåling (dark/white reference)
- Tilgjengelig fra [TCIA](https://www.cancerimagingarchive.net/) under CC BY 4.0
- Lagres lokalt i `PKG - HistologyHSI-GB/` — skal ikke endres

---

## Prosjektstruktur

```
ML_PIPELINE_G17_AITFMD/
├── configs/
│   ├── train.yaml                  # Hoved-treningskonfig (startpunkt)
│   ├── models/                     # Arkitekturkonfiger (baseline, resnet, msd_dense, ad_hybrid_sn)
│   ├── preprocessing/              # Preprosesseringskonfig (pipeline.yaml)
│   ├── grid_search/                # Konfiger for hyperparametersøk
│   └── inference/                  # Inferenskonfiger
├── data/
│   ├── raw/                        # Rådata (ENVI-kuber fra PKG-mappen)
│   ├── interim/                    # Mellomprodukt: masker, manifester, PCA-modeller
│   ├── processed/                  # Spektralt reduserte kuber klare for trening (.npy)
│   └── splits/                     # Tog/val/test-splitter per fold (JSON)
├── models/
│   ├── ae_avg3_16.pt               # Forhåndsberegnet autoencoder for inferens
│   └── pca_avg3_16.joblib          # Forhåndsberegnet PCA-modell for inferens
├── outputs/
│   ├── checkpoints/                # Beste modell-checkpoints fra enkelt-treningskjøringer
│   ├── exports/                    # ONNX-eksporter klare for SpectralAssist
│   ├── plots/                      # Trenings- og evalueringsplott
│   ├── reports/                    # Evalueringsrapporter og metrikker
│   └── logs/                       # Trenings- og inferenslogger
├── scripts/
│   ├── run_train.py                # Start enkelt treningskjøring
│   ├── run_kfold.py                # K-fold kryssvalidering
│   ├── run_eval.py                 # Evaluering på ROI-nivå
│   ├── run_eval_patch.py           # Evaluering på patch-nivå
│   ├── preprocessing/              # Forprosesseringsskript (PCA, masker, indeksering)
│   └── export/                     # ONNX-eksportskript
├── src/
│   ├── datasets/                   # Dataset-klasser (CubePatchDataset)
│   ├── models/cnn3d/               # Modellarkitekturer (baseline, resnet, msd_dense, ad_hybrid_sn)
│   ├── preprocessing/              # Kalibrering, wavelet, PCA, vevsmaskering, patching
│   ├── training/                   # Treningsloop, callbacks, early stopping
│   ├── evaluation/                 # Evalueringspipeline, metrikker, rapportering
│   ├── inference/                  # Inferenspipeline og heatmap-generering
│   └── utils/                      # Logging og hjelpefunksjoner
├── tests/                          # Enhetstester
├── requirements.txt
└── run_inference.py                # Inferens-inngangspunkt (historisk, erstattet av C#)
```

---

## Oppsett

```bash
cd ML_PIPELINE_G17_AITFMD
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kontroller at stier i `configs/preprocessing/pipeline.yaml` peker til riktig datasettlokasjon (`PKG - HistologyHSI-GB`).

---

## Dataflyten — fra råkube til trent modell

Pipelinen har fire hovedsteg. Steg 1–3 gjøres én gang og resultatet caches; steg 4 gjentas ved ny treningskjøring.

```
Rå ENVI-kuber (826 kanaler)
        │
        ▼
[1] Forprosessering
    • Kalibrering (flat-field): (raw − dark) / (white − dark)
    • Klipping til [0, 1]
    • Nabogjennomsnitt (avg3): 826 → 275 kanaler
    • Vevsmaskering: fjerner bakgrunn/glas
    • Spektral reduksjon: 275 → 16 kanaler
        │
        ├── Wavelet (db2) — anbefalt, beholder lokal spektral struktur
        └── PCA — lineær projeksjon, bakes inn i ONNX ved eksport
        │
        ▼
[2] Dataindeksering
    • CSV-manifest med kube-stier, klasselabels og maskestier
    • Pasient-stratifisert tog/val/test-split (JSON)
        │
        ▼
[3] Trening (CubePatchDataset)
    • 64×64 patcher ekstraherest tilfeldig per kube per epoke
    • Adam + cosine scheduler + early stopping
    • Mixed precision (AMP) for raskere trening
        │
        ▼
[4] ONNX-eksport
    • Checkpoint → model.onnx + manifest.json
    • Brukes direkte av SpectralAssist (C#)
```

---

## Forprosessering i detalj

Forprosessering kjøres én gang og lagrer reduserte `.npy`-kuber til `data/processed/`. Konfigurasjonen styres av `configs/preprocessing/pipeline.yaml`.

### Steg i riktig rekkefølge


| Steg                        | Hva                                                         | Hvorfor                                                                             |
| --------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Kalibrering**             | Flat-field-korreksjon med mørk- og lysreferanse             | Fjerner sensor- og belysningsstøy, normaliserer til relativ reflektans              |
| **Klipping**                | Begrens verdier til [0, 1]                                  | Fjerner outliers fra kalibrering (støy utenfor gyldig område)                       |
| **Nabogjennomsnitt (avg3)** | 826 → 275 kanaler ved glidende vindusgjennomsnitt (vindu=3) | Demper spektral støy og reduserer dimensjonalitet med minimal informasjonstap       |
| **Vevsmaskering**           | Binær maske som skiller vev fra bakgrunn                    | Sikrer at patcher inneholder nok vev; glass og tom bakgrunn ekskluderes fra trening |
| **Spektral reduksjon**      | 275 → 16 kanaler (wavelet eller PCA)                        | Gjør 3D CNN-trening praktisk mulig; reduserer minnebruk og treningstid drastisk     |
| **Patching**                | Deler kube i overlappende 64×64-patcher                     | Gjør datasett-størrelsen og batch-trening håndterbar                                |


### Kjøre forprosessering

```bash
# Bygg wavelet-datasett (anbefalt)
python scripts/preprocessing/build/build_wavelet_dataset.py \
    --config configs/preprocessing/pipeline.yaml

# Bygg PCA-datasett (krever at PCA-modell er fittet først)
python scripts/preprocessing/run/fit_pca.py --config configs/preprocessing/pipeline.yaml
python scripts/preprocessing/build/build_pca_dataset.py --config configs/preprocessing/pipeline.yaml

# Bygg vevsmaskene (kjøres automatisk av preprocessing, men kan kjøres alene)
python scripts/preprocessing/build/build_masks_from_avg3.py \
    --config configs/preprocessing/pipeline.yaml
```

### Wavelet vs. PCA


|              | Wavelet (db2)                                | PCA                                 |
| ------------ | -------------------------------------------- | ----------------------------------- |
| Metode       | Diskret wavelettransform langs spektralaksen | Lineær projeksjon til eigenvektorer |
| Beregning    | Rask, ingen fitting                          | Krever fitting på treningsdataene   |
| Egenskaper   | Bevarer lokal spektral struktur              | Global variansforklaring            |
| ONNX-eksport | Ekstern Python-preprosessering               | Bakes inn i ONNX-grafen             |
| Resultat     | **F1=0.921**                                 | F1≈0.89                             |


**Wavelet anbefales** fordi det ikke krever en separat modell og gir bedre resultater.

---

## Modellarkitekturer

Fire 3D CNN-arkitekturer er implementert i `src/models/cnn3d/`. Alle tar inn kuber av form `(B, 1, 16, 64, 64)` — batch × kanaler × spektralbånd × høyde × bredde.


| Modell            | Fil               | Beskrivelse                                                              |
| ----------------- | ----------------- | ------------------------------------------------------------------------ |
| `baseline_3dcnn`  | `baseline.py`     | Enkel stabel av 3D konvolusjonsblokker med max-pooling                   |
| `resnet_3dcnn`    | `resnet_style.py` | ResNet-inspirert med residualforbindelser og GroupNorm per stage         |
| `msd_dense_3dcnn` | `msd_dense.py`    | Multi-scale dense connections — kortslutter features på tvers av skalaer |
| `ad_hybrid_sn`    | `ad_hybrid_sn.py` | Hybrid arkitektur: separerer spektral og romlig konvolusjon              |


Modellkonfig velges via `model_config` i `configs/train.yaml`. Se `configs/models/` for arkitekturspesifikke hyperparametre (kanaler, blokker, dropout).

### Hvorfor 3D CNN?

HSI-kuber har en spektral dimensjon i tillegg til romlig høyde og bredde. En 3D konvolusjon kan lære lokale mønstre som strekker seg langs alle tre aksene simultaneously — noe som er viktig fordi tumor og frikt vev har ulike spektrale signaturer som varierer gradvis mellom bølgelengder, ikke uavhengig per kanal.

---

## Trening

### Enkelt treningskjøring

Treningskonfigen (`configs/train.yaml`) styrer datasett, modell, optimizer og treningsparametere.

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


| Felt                              | Beskrivelse                                                           |
| --------------------------------- | --------------------------------------------------------------------- |
| `model_config`                    | Sti til arkitektur-YAML (`configs/models/`)                           |
| `data.cube_root`                  | Rotmappe med reduserte `.npy`-kuber                                   |
| `data.cube_manifest_csv`          | CSV med kubeindeks og klasselabels                                    |
| `data.patch_h / patch_w`          | Patchstørrelse i piksler (må matche modellens input, vanligvis 64×64) |
| `data.patches_per_cube`           | Antall tilfeldige patcher per kube per epoke                          |
| `trainer.max_epochs`              | Maks antall treningsepoker                                            |
| `trainer.early_stopping_patience` | Stopp trening etter N epoker uten forbedring                          |
| `trainer.mixed_precision`         | Aktiver AMP for raskere trening på GPU/MPS                            |
| `optimizer.lr`                    | Læringsrate (standard: 5e-5)                                          |
| `optimizer.weight_decay`          | L2-regularisering                                                     |
| `loss.class_weighting`            | Vekt tap etter klassefordeling (anbefalt ved ubalansert data)         |


Checkpoints lagres til `outputs/checkpoints/<navn>_<dato>_best.pt`.

### K-fold kryssvalidering

K-fold splitter pasientene i K grupper og trener K separate modeller — én per fold. Dette gir et mer robust estimat på generaliseringsevnen enn én enkelt train/val-split, spesielt viktig med bare 13 pasienter.

```bash
# 5-fold kryssvalidering (standard, anbefalt)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml

# Kjør kun spesifikke folds (nyttig ved avbrutt kjøring)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml --folds 2,3,4

# Fortsett en avbrutt k-fold-kjøring (folds med fold_result.json hoppes over)
python scripts/run_kfold.py --config configs/train/wavelet/resnet.yaml \
    --resume outputs/training/kfold_resnet_3dcnn_20260419_210955
```

Resultater per fold lagres under `outputs/training/<run_name>/fold_<N>/`.

---

## Evaluering

```bash
# ROI-nivå evaluering (aggregerer patchprediksjoner per ROI med majority voting)
python scripts/run_eval.py --checkpoint outputs/checkpoints/resnet_3dcnn_best.pt \
    --config configs/train/wavelet/resnet.yaml

# Patch-nivå evaluering
python scripts/run_eval_patch.py --checkpoint outputs/checkpoints/resnet_3dcnn_best.pt \
    --config configs/train/wavelet/resnet.yaml
```

ROI-nivå-evaluering er mer klinisk relevant: en ROI klassifiseres som tumor dersom et flertall av patchene klassifiseres som tumor. Metrikker inkluderer F1, AUC, presisjon og recall.

---

## ONNX-eksport

Eksporterer et trent checkpoint til ONNX-format med tilhørende `manifest.json` for SpectralAssist.

### Obligatoriske argumenter


| Argument                  | Beskrivelse                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `--checkpoint`            | Sti til `.pt`-checkpoint                                         |
| `--out-dir`               | Utdatamappe (opprettes hvis den ikke finnes)                     |
| `--spectral-bands`        | Antall spektrale bånd **etter** reduksjon (f.eks. `16`)          |
| `--patch-h` / `--patch-w` | Patchstørrelse i piksler — må matche trening (vanligvis `64 64`) |


### Valgfrie argumenter


| Argument               | Standard                              | Beskrivelse                                                               |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| `--reducer-method`     | Fra `pipeline.yaml`                   | Overstyrer spektral reduksjonsmetode: `pca`, `wavelet`, `ae` eller `none` |
| `--raw-bands`          | —                                     | Antall råbånd FØR reduksjon (f.eks. `275`). Nødvendig for wavelet/ae      |
| `--onnx-name`          | `model.onnx`                          | Filnavn på ONNX-filen                                                     |
| `--model-config`       | Fra checkpoint                        | Override modell-YAML                                                      |
| `--pipeline-config`    | `configs/preprocessing/pipeline.yaml` | Preprosesseringskonfig som manifest leses fra                             |
| `--validation-roi-dir` | —                                     | ROI-mappe med rå ENVI-filer for integrasjonsvalidering                    |
| `--description`        | —                                     | Fritekstbeskrivelse i manifest                                            |


### Valideringskube (`--validation-roi-dir`)

Valideringskuben brukes til å verifisere at C#-pipelinen i SpectralAssist gir identisk output som Python-pipelinen. Eksport-scriptet:

1. Sliserer ut en patch fra midten av kuben
2. Kjører Python-preprosesseringspipelinen (kalibrering → klipping → avg3 → spektral reduksjon)
3. Sender patchen gjennom PyTorch-modellen og lagrer forventede logits/softmax

Dette skrives til `roi_validation/` i eksportmappen og legges inn i `manifest.json` under `validation.expected_output`.

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

---

## manifest.json — hva er det og hvorfor?

`manifest.json` er bindeleddet mellom Python-treningspipelinen og C#-applikasjonen. Den forteller SpectralAssist nøyaktig hvordan den skal forprosessere en ny kube og tolke ONNX-modellens output:


| Felt                           | Hva det betyr                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `input_spec`                   | Forventet tensorform og antall spektrale bånd modellen forventer                                          |
| `pipeline.preprocessing.steps` | Hvilke preprosesseringssteg C# skal kjøre (f.eks. `calibrate`, `clip`, `neighbor_average`, `tissue_mask`) |
| `pipeline.spectral_reducer`    | Reduksjonsmetode og om den er bakt inn i ONNX (`embedded_in_onnx: true` for PCA)                          |
| `validation.expected_output`   | Forventede logits/softmax for referansepatchen — brukes til integrasjonstest i C#                         |


---

## SpectralAssist-integrasjon

Ferdigtrente modeller lastes inn av SpectralAssist (Avalonia C#-applikasjon i `GUI_G17_AITFMD/spectral-assist/`). En eksportmappe med `model.onnx`, `manifest.json` og `roi_validation/` utgjør en komplett modellpakke som kopieres inn i applikasjonen.

For wavelet-modeller kjører C# wavelettransformen selv (basert på parametrene i manifest). For PCA-modeller er transformasjonen bakt inn i ONNX-grafen — C# trenger ikke gjøre noe ekstra.

---

## Forhåndsberegnede modeller (`models/`)

Disse brukes ved inferens og trenger ikke regenereres med mindre datasettet endres:


| Fil                  | Beskrivelse                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| `pca_avg3_16.joblib` | PCA-modell fittet på treningsdata: 16 komponenter, avg3-forprosessering |
| `ae_avg3_16.pt`      | Autoencoder-modell: 16 latente kanaler, avg3-forprosessering            |


