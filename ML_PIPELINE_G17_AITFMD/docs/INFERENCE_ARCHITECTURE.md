# Inferanse-arkitektur: GUI → Backend

Dette dokumentet beskriver grensesnittet mellom GUI og ML-backend, slik at GUI er uavhengig av modellmotoren (PyTorch nå, ONNX senere).

## Implementert (Fase 1)

- `run_inference.py` – entry point med `--input`, `--output`/`--output-dir`, `--config`
- `src/inference/pipeline.py` – preprocessing: kalibrering → clipping → avg3 → PCA16 → masking → patchifisering
- `configs/inference/default.yaml` – inference-config
- Output: `prediction.json`, `metadata.json`, `pca16_cube.npy`

**Forutsetning:** Kjør `fit_pca.py` og `build_pca_dataset.py` først for å ha PCA-modell tilgjengelig.

---

## 1. Ansvarsfordeling

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GUI (C#/Avalonia)                                                      │
│  • Sender: input-sti, config-sti, modellnavn, output-mappe               │
│  • Leser: prediction.json, metadata.json, heatmaps                        │
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
│       └── default.yaml          # Inferanse-config (modell, backend, paths)
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
| `--config` | Nei | Sti til inference-config (default: configs/inference/default.yaml) |
| `--model` | Nei | Modellnavn/sti (overstyrer config) |

**Eksempel:**
```bash
python run_inference.py \
  --input /path/to/ROI_01.hdr \
  --output-dir /path/to/results/run_001 \
  --config configs/inference/default.yaml
```

---

## 4. Kontrakt: Output (run_inference → GUI)

Alt skrives til `--output-dir`:

| Fil | Innhold |
|-----|---------|
| `prediction.json` | Hovedresultat – status, predictions, summary |
| `metadata.json` | Modellnavn, varighet, backend, versjon |
| `heatmap.npy` | (Valgfritt) Heatmap for overlay |
| `mask.npy` | (Valgfritt) Tissue mask |

### prediction.json – schema

```json
{
  "status": "ok",
  "input_path": "/path/to/input.hdr",
  "timestamp": "2025-03-06T12:00:00Z",
  "predictions": [
    { "x": 10, "y": 20, "score": 0.92, "label": "anomaly" }
  ],
  "summary": {
    "anomaly_ratio": 0.12,
    "total_pixels": 1000,
    "message": "Inference completed successfully."
  },
  "heatmap_path": "heatmap.npy",
  "mask_path": "mask.npy"
}
```

Ved feil:
```json
{
  "status": "error",
  "error": "File not found: ...",
  "timestamp": "..."
}
```

### metadata.json

```json
{
  "model": "ae16_cnn",
  "backend": "pytorch",
  "duration_ms": 1234,
  "version": "1.0"
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
# configs/inference/default.yaml
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
