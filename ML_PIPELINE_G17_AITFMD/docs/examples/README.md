# Eksempel-output fra `run_inference`

**Schema version 2** (`prediction.json` ved suksess).

| Fil | Beskrivelse |
|-----|-------------|
| `prediction_ok_example.json` | Full struktur med `input`, `model_info`, `decision`, `spatial`, `preprocessing`, `tissue_mask`, `patch_stats`, `predictions` (med `id`, `probabilities`), `summary`, `run` |
| `prediction_error_example.json` | Feilrespons |
| `prediction_tissue_mask_disabled_example.json` | Tissue mask av (forkortet tekstfelter med `…` der det er likt som OK-eksempelet) |

`patch_stats` i fiktive eksempler kan avvike fra reell kjøring; ved ekte inferanse stemmer `evaluated` med lengden på `predictions`.
