# Gjenoppbygging av reduksjonsdatasett (avg16 / PCA / wavelet / AE)

Alle fire variantene bruker nå **samme maskkonfig** i YAML (`mask.root`, `mask.require`, `mask.apply_to_cube`):

- **`apply_to_cube: true`** — bakgrunn settes til **0** på `avg3`-kuben før spektral reduksjon (eller før encoding i AE).
- **`require: true`** — bygg/fit feiler hvis `patient_id/roi_name_mask.npy` mangler under `mask.root`.

## 0. Forutsetninger

1. **avg3-kuber** på plass (`data/` eller sti i `configs/preprocessing/*.yaml` → `paths.input_root`).
2. **Patient-split** (`data/splits/patient_split.csv`) som matcher ROIs.
3. **Vevsmasker** for alle ROIs i split:  
   `mask.root / <patient_id> / <roi_name>_mask.npy`  
   Typisk generert med `scripts/preprocessing/build/build_masks_from_avg3.py` (eller `run_preprocessing.py` som maskerer fra kalibrert kube).

**Sjekk:** `mask.root` i YAML skal peke på samme mappe som du bruker i trening (`train.yaml` → `mask_root`).

## 1. avg16 (band-gjennomsnitt)

```bash
cd ML_PIPELINE_G17_AITFMD
python scripts/preprocessing/build/build_avg_baseline_dataset.py --config configs/preprocessing/avg_baseline.yaml
```

Output: `paths.output_root` + manifest (`manifest_csv`).

## 2. PCA — **fit** deretter **build**

`fit_pca.py` bruker **kun piksler der maske > 0** på train-split (ikke-bakgrunn).

```bash
python scripts/preprocessing/run/fit_pca.py --config configs/preprocessing/pca.yaml
python scripts/preprocessing/build/build_pca_dataset.py --config configs/preprocessing/pca.yaml
```

## 3. Wavelet

```bash
python scripts/preprocessing/build/build_wavelet_dataset.py --config configs/preprocessing/wavelet.yaml
```

## 4. Autoencoder — **trening** deretter **build**

Trening laster kube og maskerer ved load (samme `mask`-seksjon i `autoencoder.yaml`).

```bash
python scripts/preprocessing/run/train_autoencoder.py --config configs/preprocessing/autoencoder.yaml
python scripts/preprocessing/build/build_ae_dataset.py --config configs/preprocessing/autoencoder.yaml
```

## Viktig

- **Endre `apply_to_cube` / `require`** til `false` bare hvis du bevisst vil kjøre uten maske (ikke anbefalt for sammenlignbar pipeline).
- Etter omdreining av **PCA** eller **AE** må du **fit/trening på nytt** før `build_*` for å matche de nye maskerte dataene.
- Sett `overwrite: true` (eller slett gamle `.npy`) i respektive `runtime`/`build` når du vil tvinge full regenerering.
