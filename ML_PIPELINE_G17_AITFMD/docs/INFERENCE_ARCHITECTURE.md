# Inferanse-arkitektur: GUI → Backend

Dette dokumentet beskriver grensesnittet mellom GUI og ML-backend, slik at GUI er uavhengig av modellmotoren (PyTorch nå, ONNX senere).

## Implementert (Fase 1)

- `run_inference.py` – entry point med `--input`, `--output`/`--output-dir`, `--config`
- `src/inference/pipeline.py` – preprocessing: kalibrering → clipping → avg3 → PCA16 → masking → patchifisering
- `configs/inference/pytorch.yaml` – inference-config
- Output: `prediction.json` (**schema_version 2**), valgfritt `heatmap.png` / `.npy`, `reduced_cube.npy`

**Forutsetning:** Kjør `fit_pca.py` og `build_pca_dataset.py` først for å ha PCA-modell tilgjengelig.

---

## 1. Ansvarsfordeling

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GUI (C#/Avalonia)                                                      │
│  • Sender: input-sti, config-sti, modellnavn, output-mappe               │
│  • Leser: prediction.json (heatmap kan bygges i GUI fra predictions + spatial) │
│  • VET IKKE: PyTorch, ONNX, PCA, wavelet, preprocessing-detaljer         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  run_inference.py (nivå 1 – entry point)                                │
│  • Parser CLI-argumenter                                                 │
│  • Validerer input                                                       │
│  • Kaller pipeline                                                       │
│  • Skriver standardisert output til output-mappe                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/inference/pipeline.py (nivå 2 – orkestrering)                      │
│  • Laster config                                                         │
│  • Kjører preprocessing (calibration, spectral reduction, patching)       │
│  • Kaller model backend                                                  │
│  • Bygger resultat (predictions, heatmap)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  src/inference/backend/ (nivå 3 – modellmotoren)                        │
│  • pytorch_backend.py  → laster .pt, kjører inference                    │
│  • onnx_backend.py     → (senere) laster .onnx, kjører via ONNX Runtime │
│  • Backend velges via config; pipeline kjenner kun abstraksjonen         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mappestruktur

```
ML_PIPELINE_G17_AITFMD/
├── run_inference.py              # Entry point – kalles av GUI
├── configs/
│   └── inference/
│       └── pytorch.yaml          # Inferanse-config (modell, backend, paths)
├── src/
│   └── inference/
│       ├── __init__.py
│       ├── pipeline.py           # Orkestrering: preprocess → backend → result
│       ├── contract.py           # Output-schema (JSON-struktur)
│       └── backend/
│           ├── __init__.py
│           ├── base.py           # Abstrakt backend-interface
│           └── pytorch_backend.py
├── models/                       # (eller data/interim/ae_models etc.)
│   └── ...                       # .pt, .onnx – lastes av backend
└── docs/
    └── INFERENCE_ARCHITECTURE.md
```

---

## 3. Kontrakt: CLI-argumenter (GUI → run_inference)

| Argument | Påkrevd | Beskrivelse |
|----------|---------|-------------|
| `--input` | Ja | Sti til input (f.eks. .hdr eller .npy) |
| `--output-dir` | Ja | Mappe der resultater skrives |
| `--config` | Nei | Sti til inference-config (default: configs/inference/pytorch.yaml) |
| `--model` | Nei | Modellnavn/sti (overstyrer config) |

**Eksempel:**
```bash
python run_inference.py \
  --input /path/to/ROI_01.hdr \
  --output-dir /path/to/results/run_001 \
  --config configs/inference/pytorch.yaml
```

---

## 4. Kontrakt: Output (run_inference → GUI)

Alt skrives til `--output-dir`:

| Fil | Innhold |
|-----|---------|
| `prediction.json` | **Eneste obligatoriske JSON** – se schema under |
| `heatmap.png` / `heatmap.npy` | (Valgfritt) `write_heatmap_assets: true` i inference-config – ellers ikke skrevet |
| `reduced_cube.npy` | (Valgfritt) `write_reduced_cube: true` – debugging |

### prediction.json – schema (**schema_version 2**)

Fullt eksempel: [`examples/prediction_ok_example.json`](examples/prediction_ok_example.json).

Hovedblokker ved `status: ok`:

| Blokk | Innhold |
|--------|---------|
| `input` | `path`, `timestamp` |
| `model_info` | `name`, `backend`, `checkpoint_path`, `checkpoint_file`, `model_config_path`, `num_classes`, `class_names` |
| `decision` | `positive_class`, `anomaly_threshold`, `score_type`, `description`, `applies_to_class_index` (pytorch: softmax sannsynlighet for positiv klasse) |
| `spatial` | `cube_shape`, `patch_*`, `stride_*`, `patch_anchor`, `origin`, `coordinate_space`, `coordinate_space_description`, `axes_order` |
| `preprocessing` | `pipeline_config`, `steps` (enabled per steg), `spectral_reducer`, `num_spectral_bands`, `min_tissue_ratio_patch` |
| `tissue_mask` | ROI-nivå maskestatistikk (etter avg3, før spektral reduksjon) |
| `patch_stats` | `total_possible`, `evaluated`, `filtered_by_tissue`, `description` |
| `predictions` | Sparse liste: `id`, `y`, `x`, `score`, `probabilities` (per klasse ved binær), `label` |
| `summary` | `anomaly_ratio`, `message` |
| `run` | `duration_ms` |

**Koordinater:** `(y, x)` er **øvre venstre** hjørne av patchen i **`reduced_cube`** (ikke rå HDR). Se `spatial.coordinate_space_description`.

**Heatmap i GUI:** aggreger `predictions[].score` over overlapp med samme logikk som `src/inference/heatmap.py` (`build_heatmap`).

- **`tissue_mask.background_percent`**: andel piksler som **bakgrunn** på hele ROI etter avg3. **`patch_stats.filtered_by_tissue`**: patches hoppet pga. for lav vev-andel i patch-vindu (`min_tissue_ratio_patch`).

Terskel for `label` styres av `decision.anomaly_threshold` i `configs/inference/pytorch.yaml` (nøkkel `decision.anomaly_threshold`).

Ved feil:
```json
{
  "schema_version": 2,
  "status": "error",
  "error": "File not found: ...",
  "timestamp": "..."
}
```

---

## 5. Backend-interface (abstraksjon)

```python
# src/inference/backend/base.py
class InferenceBackend(Protocol):
    def load(self, model_path: Path, config: dict) -> None: ...
    def predict(self, patches: np.ndarray) -> np.ndarray: ...
```

Pipeline velger backend ut fra config:
```yaml
# configs/inference/pytorch.yaml
backend: pytorch   # eller onnx
model_path: "models/ae16_cnn.pt"
```

---

## 6. Migrasjon: PyTorch → ONNX

1. **Nå:** `pytorch_backend.py` laster `.pt`, kjører `model(patches)`.
2. **Senere:** Legg til `onnx_backend.py` som laster `.onnx`, bruker `onnxruntime`.
3. **Config:** Bytt `backend: onnx` og `model_path: "models/ae16_cnn.onnx"`.
4. **GUI:** Ingen endring – leser fortsatt `prediction.json` fra `--output-dir`.

---

## 7. GUI-integrasjon (C#)

```csharp
// InferenceRunner.cs – eksempel
var outputDir = Path.Combine(Path.GetTempPath(), $"inference_{Guid.NewGuid():N}");
Directory.CreateDirectory(outputDir);

var args = $"\"{scriptPath}\" --input \"{hdrPath}\" --output-dir \"{outputDir}\"";
// ... Process.Start ...

var predictionPath = Path.Combine(outputDir, "prediction.json");
var json = File.ReadAllText(predictionPath);
var result = JsonSerializer.Deserialize<InferenceResult>(json);

// Vis heatmap fra Path.Combine(outputDir, "heatmap.npy") om ønskelig
```

GUI trenger kun å kjenne `prediction.json`-schema og stier til eventuelle overlay-filer.
