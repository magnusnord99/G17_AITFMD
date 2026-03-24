# 3D-CNN → ONNX → SpectralAssist (steg for steg)

Dette dokumentet følger én linje: **PyTorch-trening** → **eksport** → **kopier til GUI** → **C#-inferens**.

## Steg 0: Forstå tensor-form (samme som `pytorch_backend.py`)

Trening og inferens bruker **patch-tensor**:

| Beskrivelse | Shape |
|-------------|--------|
| Rå patch (NumPy) | `(N, H, W, C)` med `C` = spektrale bånd (f.eks. 16) |
| Etter `permute` + `unsqueeze` | `(N, 1, C, H, W)` |
| PyTorch `Conv3d` | `(B, C_in, D, H, W)` = **`(batch, 1, spektral_dybde, H, W)`** |

Her er **D = antall bånd** (f.eks. 16), **ikke** en ekstra «RGB-kanal» — `in_channels=1` betyr én volumetrisk kanal inn i 3D-conv.

**Konsekvens for C#:** Etter preprocessing har du `HsiCube` med BSQ-data; vi bygger ONNX-input **`[1, 1, C, patchH, patchW]`** (samme minnelayout som trening: spektral «dybde» langs tredje akse).

---

## Steg 1: Tren modell og velg checkpoint (`.pt`)

Når du er fornøyd med validering, noter:

- sti til **checkpoint** (inneholder `model_state_dict`, helst `model_config_path`),
- **YAML** som beskriver arkitektur (f.eks. `configs/models/baseline_3dcnn.yaml`),
- **C** (spektrale bånd etter din pipeline, ofte **16**),
- **patch_h / patch_w** (må være de samme som ved trening, f.eks. 64×64).

---

## Steg 2: Eksporter ONNX + manifest (fra ML-mappen)

Aktiver venv, kjør fra `ML_PIPELINE_G17_AITFMD`:

```bash
source .venv/bin/activate
python scripts/export_cnn3d_onnx.py \
  --checkpoint path/til/best.pt \
  --model-config configs/models/baseline_3dcnn.yaml \
  --spectral-bands 16 \
  --patch-h 64 \
  --patch-w 64 \
  --out-dir outputs/onnx_cnn3d_release_v1
```

Dette skriver:

- `model.onnx`
- `manifest.json` (med `input_rank: 5` og `input_shape` for NCDHW)

---

## Steg 3: Kopier pakke inn i SpectralAssist (det som shipper)

Kopier **hele** output-mappen (minst `model.onnx` + `manifest.json`, ev. `.onnx.data` hvis ORT lager den) til f.eks.:

`GUI_G17_AITFMD/spectral-assist/SpectralAssist/Assets/models/<navn>/`

Oppdater `manifest.json` feltet `artifacts.pipeline_onnx` hvis du bruker annet filnavn enn `model.onnx`.

**Eksempel i repo:** `baseline_3dcnn_20260324_083658_last` (16 bånd PCA, patch 32×32) ligger under `Assets/models/` etter eksport.

---

## Steg 4: C# — bruk `Onnx3DCnnClassifier`

- Last pakke med `ModelLoader.LoadPackage(mappeMedManifest)`.
- Bruk **`Onnx3DCnnClassifier`** (ikke den gamle `OnnxClassifier` for 4D) når manifest har **`input_rank: 5`**.
- `HsiCube` må ha **C** bånd og patch-størrelse som i manifest.

---

## Steg 5: Når skal du eksportere på nytt?

Kun når du **fryser en ny modell** til bruk i app (ny checkpoint / ny arkitektur / andre patch-dimensjoner) — ikke etter hver epoch.
