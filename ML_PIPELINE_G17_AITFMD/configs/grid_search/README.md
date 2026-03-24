# Grid search (overnight tuning)

**Ferdige grids**

| Fil | Formål |
|-----|--------|
| `nightly.yaml` | Smal grid (lr × wd) — kort referanse |
| `low_lr_grid.yaml` | **12 runs**: lav `optimizer.lr` (1e-5…3e-4) × `weight_decay` — bruk når høy lr ikke fungerer |

1. **Rediger** valgfri YAML under `configs/grid_search/`, eller bruk `low_lr_grid.yaml` som den er.
2. **Sjekk antall kombinasjoner:**
   ```bash
   python scripts/grid_search_train.py --grid configs/grid_search/nightly.yaml --dry-run
   ```
3. **Kjør** (sekvensielt — én trening av gangen, best for MPS/GPU):
   ```bash
   cd ML_PIPELINE_G17_AITFMD
   python scripts/grid_search_train.py --grid configs/grid_search/nightly.yaml
   ```

**Output**

- `outputs/grid_search/configs_<timestamp>/` — genererte `train.yaml` per kjøring  
- `outputs/grid_search/summary_<timestamp>.csv` — én rad per run: exit code, tid, `best_val_loss` / `best_epoch` fra nyeste `train_report_*.json`, alle grid-parametre  

Grid-scriptet **fanger ikke stdout** fra `run_train`, så du ser **tqdm og epoker live** (som ved direkte kjøring). Uten det ville terminalen være tom i timevis.

**Tips**

- Hold `seed` lik i `train.yaml` når du sammenligner.  
- Flere dimensjoner i `grid` = eksponentielt flere runs — start smalt (lr + wd).  
- Test med `--max-runs 1` før du starter natten.
