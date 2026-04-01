# Claude Code — prosjektkontekst (G17 / HistologyHSI-GB 3D CNN)

## Rot og miljø

- Arbeidsrot for ML-koden er denne mappen (`ML_PIPELINE_G17_AITFMD`).
- Bruk prosjektets venv før `python`/`pip`: `source .venv/bin/activate` (se `README.md`).
- Kildekode: `src/` (modeller under `src/models/cnn3d/`, trening/eval under `src/training/`, `src/evaluation/`).

## Viktige configs

- Trening: `configs/train.yaml` (+ `configs/models/baseline_3dcnn.yaml` for arkitektur).
- Preprocessing / PCA: `configs/preprocessing/pca.yaml`, `scripts/preprocessing/run/fit_pca.py`.
- Grid search: `configs/grid_search/`, `scripts/grid_search_train.py`.

## Vanlige kommandoer

- Trening: `python scripts/run_train.py --config configs/train.yaml`
- Evaluering: `scripts/run_eval.py` (se `configs/` og `docs/`).

## Data og artefakter

- `data/` (raw/interim/processed) og `outputs/` (checkpoints, plots, reports) er typisk ikke i git; ikke commit store binærfiler uten avtale.
- PCA-modeller for inferens: `models/` (se `models/README.md`).

## Stil

- Følg eksisterende mønstre i repo; unngå unødvendige refaktorer utenfor oppgaven.
