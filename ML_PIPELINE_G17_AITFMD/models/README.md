# Modeller for inferanse

Denne mappen inneholder trente modeller som brukes av `run_inference.py`. De er inkludert i repo for at kollegaer kan bruke pipelinen uten å trene selv.

| Fil | Beskrivelse | Brukes når |
|-----|-------------|------------|
| `pca_avg3_16.joblib` | PCA for spektral reduksjon (275 → 16 kanaler) | `spectral_reducer: "pca"` i inference-config |
| `ae_avg3_16.pt` | Autoencoder for spektral reduksjon | `spectral_reducer: "ae"` i inference-config |

**Reproduksjon:** Modellene trenes med:
- PCA: `python scripts/preprocessing/run/fit_pca.py --config configs/preprocessing/pca.yaml`
- AE: `python scripts/preprocessing/run/train_autoencoder.py` (med riktig config)
