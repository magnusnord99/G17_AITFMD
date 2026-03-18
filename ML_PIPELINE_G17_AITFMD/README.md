# HistologyHSI-GB 3D CNN Pipeline

This project structure is organized for a clean workflow:

1. Preprocessing ENVI hyperspectral cubes.
2. Generating tissue-aware 3D patches.
3. Training and evaluating a 3D CNN classifier.

## Project Layout

```text
hsi_3dcnn_project/
  configs/
  models/              # PCA + AE for inference (committed for colleagues)
  data/
    raw/
    interim/
    processed/
    splits/
  docs/
  notebooks/
  outputs/
    checkpoints/
    figures/
    logs/
    reports/
  scripts/
  src/
    preprocessing/
    datasets/
    models/
      cnn3d/
    training/
    evaluation/
    utils/
```

## Notes

- Keep your original dataset in `PKG - HistologyHSI-GB` unchanged.
- Point the preprocessing config to that folder as input.
- Write only derived artifacts into `hsi_3dcnn_project/data/*` and `outputs/*`.

## Team Setup (Git + Colleague)

Use the same steps on both machines for reproducible runs.

1. Clone repo and enter project folder.
2. Create and activate virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux)
3. Install dependencies:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
4. Verify paths in `configs/preprocessing/preprocessing.yaml` match local dataset location.
5. **Inference:** `models/` contains PCA and AE models. Run `run_inference.py --input <ROI> --output-dir <dir>` – no extra setup needed.

### Recommended Git Practice

- Do not commit `.venv`, generated patches, checkpoints, logs, or derived data.
- Commit code, configs, lightweight metadata, and `models/` (PCA + AE for inference).
- Keep branch names task-based (example: `feat/preprocessing-indexer`).
