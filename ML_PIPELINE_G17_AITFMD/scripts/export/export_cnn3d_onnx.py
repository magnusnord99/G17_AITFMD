#!/usr/bin/env python3
"""
Eksporter trent 3D-CNN til ONNX + manifest.json for SpectralAssist (NCDHW).

- Preprocessing (manifest): configs/preprocessing/pipeline.yaml
- **Reducer**: fra `spectral_reduction.reducer` i samme YAML hvis `--reducer-method` utelates.
- **PCA**: `embedded_in_onnx` true; PCA (joblib) + CNN i én ONNX; input (1,1,n_raw,H,W).
- **Annen reducer**: kun CNN; `embedded_in_onnx` false.

Se docs/CNN3D_ONNX_WORKFLOW.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import build_model_from_config

DEFAULT_PIPELINE_YAML = PROJECT_ROOT / "configs" / "preprocessing" / "pipeline.yaml"


def _step_enabled(section: dict[str, Any] | None) -> bool:
    if not section:
        return False
    return bool(section.get("enabled", True))


def _load_pipeline_dict(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw.get("pipeline") or {}


def spectral_from_pipeline(pipeline: dict[str, Any]) -> tuple[str, Path | None]:
    """
    Returner (reducer_method, pca_model_path_eller_none).
    PCA-path settes kun når reducer er pca og sti finnes.
    """
    sr = pipeline.get("spectral_reduction") or {}
    if not _step_enabled(sr):
        return "none", None
    r = str(sr.get("reducer", "none")).lower().strip()
    pca_path: Path | None = None
    if r == "pca":
        rel = sr.get("pca_model")
        if not rel:
            raise ValueError("pipeline.spectral_reduction.reducer er pca men pca_model mangler")
        pca_path = (PROJECT_ROOT / str(rel)).resolve()
    return r, pca_path


def preprocessing_from_pipeline_yaml(path: Path) -> dict[str, Any]:
    p = _load_pipeline_dict(path)
    steps: list[str] = []
    params: dict[str, Any] = {
        "band_reduce_out_bands": 0,
        "band_reduce_strategy": "",
    }

    if _step_enabled(p.get("calibration")):
        steps.append("calibrate")
        c = p["calibration"]
        params["calibration_epsilon"] = float(c.get("eps", 1e-8))

    if _step_enabled(p.get("clip")):
        steps.append("clip")
        c = p["clip"]
        params["clip_min"] = float(c.get("clip_min", 0.0))
        params["clip_max"] = float(c.get("clip_max", 1.0))

    if _step_enabled(p.get("avg3")):
        steps.append("neighbor_average")
        a = p["avg3"]
        params["neighbor_average_window"] = int(a.get("reduction_window", 3))

    if _step_enabled(p.get("tissue_mask")):
        steps.append("tissue_mask")
        t = p["tissue_mask"]
        params["tissue_mask_method"] = str(t.get("method", "mean_std_percentile"))
        params["tissue_mask_q_mean"] = float(t.get("q_mean", 0.5))
        params["tissue_mask_q_std"] = float(t.get("q_std", 0.4))
        params["tissue_mask_min_object_size"] = int(t.get("min_object_size", 1000))
        params["tissue_mask_min_hole_size"] = int(t.get("min_hole_size", 1000))
        if "tissue_side" in t:
            params["tissue_mask_tissue_side"] = str(t["tissue_side"])

    sr = p.get("spectral_reduction") or {}
    if _step_enabled(sr):
        reducer = str(sr.get("reducer", "none")).lower().strip()
        if reducer != "none":
            steps.append(reducer)
            params["band_reduce_strategy"] = reducer
            if reducer == "wavelet":
                wav = sr.get("wavelet") or {}
                params["band_reduce_out_bands"] = int(wav.get("target_bands", 0))

    return {"steps": steps, "params": params}


def _sync_preprocessing_reducer(
    block: dict[str, Any],
    reducer_method: str,
    spectral_bands: int | None,
) -> None:
    """Oppdater preprocessing-blokken in-place slik at steps og params
    reflekterer den faktiske reducer_method (ikke pipeline.yaml-default).

    Kaller på når --reducer-method overstyrer hva YAML sier.
    """
    KNOWN_REDUCERS = {"pca", "wavelet", "ae", "none"}
    steps: list[str] = block.get("steps", [])
    params: dict[str, Any] = block.get("params", {})

    # Fjern eventuelle eksisterende reducer-steg fra steps
    steps[:] = [s for s in steps if s not in KNOWN_REDUCERS]

    if reducer_method and reducer_method != "none":
        steps.append(reducer_method)
        params["band_reduce_strategy"] = reducer_method
        if spectral_bands is not None:
            params["band_reduce_out_bands"] = spectral_bands
        elif reducer_method != "wavelet":
            params["band_reduce_out_bands"] = 0
    else:
        params["band_reduce_strategy"] = ""
        params["band_reduce_out_bands"] = 0


class TorchSpectralPCA(nn.Module):
    """(B,1,n_in,H,W) -> (B,1,n_out,H,W), ekvivalent sklearn PCA.transform per piksel."""

    def __init__(self, pca: Any) -> None:
        super().__init__()
        n_in = int(pca.n_features_in_)
        n_out = int(pca.n_components_)
        mean = torch.from_numpy(pca.mean_.astype("float32"))
        comp = torch.from_numpy(pca.components_.astype("float32"))
        self.register_buffer("mean", mean)
        self.register_buffer("components_t", comp.T.contiguous())
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, nin, h, w = x.shape
        if c != 1:
            raise ValueError(f"Forvent (B,1,n_in,H,W), fikk c={c}")
        if nin != self.n_in:
            raise ValueError(f"Forvent n_in={self.n_in}, fikk {nin}")
        x_hw = x.squeeze(1).permute(0, 2, 3, 1)
        flat = x_hw.reshape(-1, nin)
        out = (flat - self.mean) @ self.components_t
        out = out.reshape(b, h, w, self.n_out).permute(0, 3, 1, 2).unsqueeze(1)
        return out


class PCAThenCNN(nn.Module):
    def __init__(self, pca: TorchSpectralPCA, cnn: nn.Module) -> None:
        super().__init__()
        self.pca = pca
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(self.pca(x))


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'")
    return ckpt


def _layer_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, m in model.named_modules():
        if not name:
            continue
        if len(list(m.children())) > 0:
            continue
        cls = m.__class__.__name__
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _load_model_yaml(cfg_path: Path) -> dict[str, Any]:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _resolve_path_from_project(raw: str | None) -> Path | None:
    if raw is None:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pick_history_row(ckpt: dict[str, Any]) -> dict[str, Any] | None:
    history = ckpt.get("history")
    if not isinstance(history, list) or not history:
        return None
    rows = [row for row in history if isinstance(row, dict)]
    if not rows:
        return None
    best_epoch = ckpt.get("best_epoch")
    if isinstance(best_epoch, int):
        for row in rows:
            if row.get("epoch") == best_epoch:
                return row
    return rows[-1]


def _derive_dataset_label(
    train_cfg: dict[str, Any] | None,
    train_report: dict[str, Any] | None,
) -> str | None:
    if isinstance(train_cfg, dict):
        data = train_cfg.get("data") or {}
        explicit = str(data.get("dataset_name", "")).strip()
        if explicit:
            return explicit
        cube_root = str(data.get("cube_root", "")).strip()
        if cube_root:
            root_name = Path(cube_root).name
            if root_name:
                return root_name
        manifest_csv = str(data.get("cube_manifest_csv", "")).strip()
        if manifest_csv:
            return Path(manifest_csv).stem

    if isinstance(train_report, dict):
        manifest_path = str(train_report.get("manifest_path", "")).strip()
        if manifest_path:
            return Path(manifest_path).stem
    return None


def _resolve_training_for_manifest(
    *,
    ckpt: dict[str, Any],
    train_report: dict[str, Any] | None,
    train_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics: dict[str, float | None] = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    extra_metric_keys = (
        "val_auc_roi",
        "val_avg_precision_roi",
        "val_f1_at_0.5_roi",
        "val_f1_at_opt_roi",
        "val_threshold_opt_roi",
        "auc_roc",
    )

    row = _pick_history_row(ckpt)
    if row is not None:
        metrics["accuracy"] = _to_float_or_none(row.get("val_acc"))
        metrics["precision"] = _to_float_or_none(row.get("val_precision"))
        metrics["recall"] = _to_float_or_none(row.get("val_recall"))
        metrics["f1"] = _to_float_or_none(row.get("val_f1"))
        for k in extra_metric_keys:
            v = _to_float_or_none(row.get(k))
            if v is not None:
                metrics[k] = v
        # Eldre history-rader lagrer auc_roc kun inni val_metrics (dict eller streng).
        # Hent manglende nøkler derfra.
        raw_vm = row.get("val_metrics")
        if isinstance(raw_vm, str):
            import ast
            try:
                raw_vm = ast.literal_eval(raw_vm)
            except (ValueError, SyntaxError):
                raw_vm = None
        if isinstance(raw_vm, dict):
            for k in extra_metric_keys:
                if k not in metrics or metrics[k] is None:
                    v = _to_float_or_none(raw_vm.get(k))
                    if v is not None:
                        metrics[k] = v
    else:
        val_metrics = ckpt.get("val_metrics")
        if isinstance(val_metrics, dict):
            metrics["accuracy"] = _to_float_or_none(val_metrics.get("accuracy"))
            metrics["precision"] = _to_float_or_none(val_metrics.get("precision"))
            metrics["recall"] = _to_float_or_none(val_metrics.get("recall"))
            metrics["f1"] = _to_float_or_none(val_metrics.get("f1"))
            for k in extra_metric_keys:
                v = _to_float_or_none(val_metrics.get(k))
                if v is not None:
                    metrics[k] = v

    samples: int | None = None
    if isinstance(train_report, dict):
        raw_samples = train_report.get("train_samples")
        if isinstance(raw_samples, int):
            samples = raw_samples
        else:
            try:
                if raw_samples is not None:
                    samples = int(raw_samples)
            except (TypeError, ValueError):
                samples = None

    training: dict[str, Any] = {
        "dataset": _derive_dataset_label(train_cfg, train_report),
        "samples": samples,
        "epochs": ckpt.get("epoch"),
        "metrics": metrics,
    }
    if isinstance(ckpt.get("best_epoch"), int):
        training["best_epoch"] = ckpt["best_epoch"]
    return training


def _build_gui_manifest(
    *,
    base: dict[str, Any] | None,
    preprocessing: dict[str, Any],
    model: torch.nn.Module,
    layers_model: torch.nn.Module | None,
    model_yaml: dict[str, Any],
    cfg_path: Path,
    ckpt: dict[str, Any],
    input_spec_bands: int,
    patch_h: int,
    patch_w: int,
    onnx_name: str,
    description: str | None,
    training: dict[str, Any],
    reducer_method: str,
    embedded_reducer_in_onnx: bool,
    reducer_input_bands: int,
    reducer_output_bands: int,
    validation_tolerance: float = 1e-3,
) -> dict[str, Any]:
    arch = model_yaml.get("architecture") or {}
    model_block = model_yaml.get("model") or {}
    arch_name = str(model_block.get("name") or cfg_path.stem)
    num_classes = int(arch.get("num_classes", 2))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_src = layers_model if layers_model is not None else model

    if isinstance(ckpt.get("classes"), list) and ckpt["classes"]:
        class_names = [str(x) for x in ckpt["classes"]]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    desc = description or (
        f"3D CNN ONNX export ({arch_name}); NCDHW in [1,1,{input_spec_bands},{patch_h},{patch_w}]"
    )

    if base is not None:
        out = copy.deepcopy(base)
        pipe = out.setdefault("pipeline", {})
        if "preprocessing" not in pipe:
            pipe["preprocessing"] = copy.deepcopy(preprocessing)
    else:
        out = {}
        out["pipeline"] = {"preprocessing": copy.deepcopy(preprocessing)}
        pipe = out["pipeline"]

    meta = out.setdefault("metadata", {})
    meta["name"] = arch_name
    meta["version"] = meta.get("version", "1.0.0")
    meta["created"] = created
    meta["author"] = meta.get("author", "ML Pipeline")
    meta["description"] = desc

    pipe["spectral_reducer"] = {
        "method": reducer_method,
        "embedded_in_onnx": embedded_reducer_in_onnx,
        "input_bands": reducer_input_bands,
        "output_bands": reducer_output_bands,
    }
    pipe["model"] = {
        "architecture": arch_name,
        "task": "classification",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layers": _layer_counts(layer_src),
    }

    out["input_spec"] = {
        "input_rank": 5,
        "tensor_layout": "NCDHW",
        "input_shape": [1, 1, input_spec_bands, patch_h, patch_w],
        "spectral_bands": input_spec_bands,
        "spatial_patch_size": [patch_h, patch_w],
        "default_stride": [patch_h // 2, patch_w // 2],
        "dtype": "float32",
    }
    out["output_spec"] = {
        "type": "logits",
        "num_classes": num_classes,
        "classes": class_names[:num_classes] if len(class_names) >= num_classes else class_names,
    }
    out["training"] = copy.deepcopy(training)
    out["artifacts"] = {"model_onnx": onnx_name, "architecture_diagram": None}
    out["validation"] = {
        "status": "pending",
        "tolerance": validation_tolerance,
        "roi_dir": None,
        "patch_coords": None,
        "expected_output": None,
    }

    return {
        "schema_version": out.get("schema_version", "1.0"),
        "metadata": out["metadata"],
        "input_spec": out["input_spec"],
        "output_spec": out["output_spec"],
        "pipeline": out["pipeline"],
        "training": out["training"],
        "artifacts": out["artifacts"],
        "validation": out["validation"],
    }


def _slice_envi_to_disk(
    src_hdr: Path,
    src_bin: Path,
    patch_y: int,
    patch_x: int,
    patch_h: int,
    patch_w: int,
    out_dir: Path,
    name: str,
) -> None:
    """
    Slice an ENVI file spatially and write to out_dir/{name}.hdr + out_dir/{name}.

    Preserves original data type, interleave, and wavelength metadata from the source
    header. Dark/white references typically have lines=1 and are copied as-is.
    """
    import spectral
    import spectral.io.envi as envi

    img = spectral.envi.open(str(src_hdr), str(src_bin))
    src_lines = int(img.metadata.get("lines", img.nrows))

    if src_lines == 1:
        # Single-line reference — slice samples (x-axis) only, keep single line.
        cols = list(range(patch_x, patch_x + patch_w))
        data = img.read_subimage([0], cols)  # (1, patch_w, bands)
        meta = dict(img.metadata)
        meta["lines"] = 1
        meta["samples"] = patch_w
        meta["header offset"] = 0
        meta.pop("description", None)
        hdr_path = out_dir / f"{name}.hdr"
        envi.save_image(str(hdr_path), data, metadata=meta, force=True, ext="")
        return

    rows = list(range(patch_y, patch_y + patch_h))
    cols = list(range(patch_x, patch_x + patch_w))
    data = img.read_subimage(rows, cols)  # (patch_h, patch_w, bands), native dtype

    meta = dict(img.metadata)
    meta["lines"] = patch_h
    meta["samples"] = patch_w
    meta["header offset"] = 0
    meta.pop("description", None)

    hdr_path = out_dir / f"{name}.hdr"
    envi.save_image(str(hdr_path), data, metadata=meta, force=True, ext="")


def _pick_patch_center(cube_h: int, cube_w: int, patch_h: int, patch_w: int) -> tuple[int, int]:
    """Return top-left (y, x) so patch is centred in the cube."""
    y = max(0, (cube_h - patch_h) // 2)
    x = max(0, (cube_w - patch_w) // 2)
    return y, x


def _build_validation_roi(
    *,
    roi_dir: Path,
    patch_y: int | None,
    patch_x: int | None,
    patch_h: int,
    patch_w: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Slice raw spatially and copy dark/white as-is into out_dir/roi_validation/."""
    import spectral
    from src.preprocessing.calibrateClip import load_envi_cube

    files = {
        "raw.hdr": roi_dir / "raw.hdr",
        "raw": roi_dir / "raw",
        "darkReference.hdr": roi_dir / "darkReference.hdr",
        "darkReference": roi_dir / "darkReference",
        "whiteReference.hdr": roi_dir / "whiteReference.hdr",
        "whiteReference": roi_dir / "whiteReference",
    }
    for label, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Validation ROI missing file '{label}': {p}")

    raw_img = spectral.envi.open(str(roi_dir / "raw.hdr"), str(roi_dir / "raw"))
    cube_h = int(raw_img.metadata.get("lines", raw_img.nrows))
    cube_w = int(raw_img.metadata.get("samples", raw_img.ncols))

    if patch_y is None or patch_x is None:
        patch_y, patch_x = _pick_patch_center(cube_h, cube_w, patch_h, patch_w)

    if patch_y + patch_h > cube_h or patch_x + patch_w > cube_w:
        raise ValueError(
            f"Patch [{patch_y}:{patch_y + patch_h}, {patch_x}:{patch_x + patch_w}] "
            f"out of bounds for cube shape ({cube_h}, {cube_w})"
        )

    val_dir = out_dir / "roi_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    for name in ("raw", "darkReference", "whiteReference"):
        _slice_envi_to_disk(
            roi_dir / f"{name}.hdr",
            roi_dir / name,
            patch_y, patch_x, patch_h, patch_w,
            val_dir, name,
        )

    # Load float32 cubes for compute_expected_output (calibration needs all three)
    raw_cube = load_envi_cube(roi_dir / "raw.hdr", roi_dir / "raw")
    dark_cube = load_envi_cube(roi_dir / "darkReference.hdr", roi_dir / "darkReference")
    white_cube = load_envi_cube(roi_dir / "whiteReference.hdr", roi_dir / "whiteReference")

    return {
        "source_roi": str(roi_dir.resolve()),
        "patch_y": patch_y,
        "patch_x": patch_x,
        "raw_cube": raw_cube,
        "dark_cube": dark_cube,
        "white_cube": white_cube,
    }


def _compute_expected_output(
    *,
    raw_cube: np.ndarray,
    dark_cube: np.ndarray,
    white_cube: np.ndarray,
    patch_y: int,
    patch_x: int,
    patch_h: int,
    patch_w: int,
    model: torch.nn.Module,
    reducer_method: str,
    spectral_bands: int,
    pipeline: dict[str, Any],
    sk_pca: Any | None,
) -> dict[str, Any]:
    """Run Python preprocessing pipeline on slice → PyTorch model → return expected output."""
    from src.preprocessing.calibrateClip import calibrate_cube, clip_cube
    from src.preprocessing.spectral_transform import reduce_bands_neighbor_average

    cal = pipeline.get("calibration") or {}
    clp = pipeline.get("clip") or {}
    avg = pipeline.get("avg3") or {}

    calibrated = calibrate_cube(raw_cube, dark_cube, white_cube, eps=float(cal.get("eps", 1e-8)))
    clipped = clip_cube(calibrated, float(clp.get("clip_min", 0.0)), float(clp.get("clip_max", 1.0)))
    reduced_bands = reduce_bands_neighbor_average(clipped, window=int(avg.get("reduction_window", 3)))

    patch = reduced_bands[patch_y : patch_y + patch_h, patch_x : patch_x + patch_w, :]

    if reducer_method == "wavelet":
        from src.preprocessing.wavelet import reduce_cube_wavelet_approx_padded

        patch = reduce_cube_wavelet_approx_padded(patch, target_bands=spectral_bands)
    elif reducer_method == "pca":
        from src.preprocessing.pca import transform_cube_with_pca

        if sk_pca is None:
            raise ValueError("sk_pca required for reducer_method='pca'")
        patch = transform_cube_with_pca(patch, sk_pca)
    elif reducer_method != "none":
        raise ValueError(f"Unknown reducer_method: {reducer_method!r}")

    # (H, W, C) → (1, 1, C, H, W)
    tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)

    softmax = F.softmax(logits, dim=1)
    return {
        "logits": [round(float(v), 6) for v in logits[0].tolist()],
        "softmax": [round(float(v), 6) for v in softmax[0].tolist()],
        "predicted_class": int(logits[0].argmax().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3D CNN to ONNX + manifest for C#.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model YAML (default: ckpt['model_config_path'] if present)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for .onnx + manifest")
    parser.add_argument(
        "--spectral-bands",
        type=int,
        required=True,
        help="CNN spektral dybde D etter reduksjon (f.eks. 16). Ved PCA-eksport må være lik PCA n_components.",
    )
    parser.add_argument("--patch-h", type=int, required=True, help="Patch height (must match training)")
    parser.add_argument("--patch-w", type=int, required=True, help="Patch width (must match training)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--onnx-name", type=str, default="model.onnx", help="Output ONNX filename")
    parser.add_argument(
        "--manifest-template",
        type=Path,
        default=None,
        help="Valgfri JSON som startpunkt (deepcopy); felt overskrives.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_YAML,
        help="preprocessing + spectral_reduction (default: configs/preprocessing/pipeline.yaml)",
    )
    parser.add_argument("--description", type=str, default=None, help="metadata.description")
    parser.add_argument("--dataset", type=str, default=None, help="training.dataset")
    parser.add_argument("--train-samples", type=int, default=None, help="training.samples")
    parser.add_argument(
        "--train-report",
        type=Path,
        default=None,
        help="Valgfri sti til train_report.json (default: ved siden av checkpoint hvis den finnes)",
    )
    parser.add_argument(
        "--reducer-method",
        type=str,
        default=None,
        help="Overstyr spectral_reduction.reducer (pca|wavelet|ae|none). Standard: fra pipeline.yaml",
    )
    parser.add_argument(
        "--validation-roi-dir",
        type=Path,
        default=None,
        help="ROI-mappe med raw/dark/white ENVI-filer. Slicer ut en patch og skriver til roi_validation/",
    )
    parser.add_argument(
        "--validation-patch-y",
        type=int,
        default=None,
        help="Øvre venstre y for valideringspatch (default: auto = midtpunkt av cube)",
    )
    parser.add_argument(
        "--validation-patch-x",
        type=int,
        default=None,
        help="Øvre venstre x for valideringspatch (default: auto = midtpunkt av cube)",
    )
    parser.add_argument(
        "--raw-bands",
        type=int,
        default=None,
        help="Antall rå spektrale bånd FØR reduksjon (f.eks. 275). Brukes for ikke-embedded reducere "
             "for å sette spectral_reducer_input_bands og neighbor_average_out_bands korrekt. "
             "Utelates automatisk for PCA (leses fra PCA-modellen).",
    )
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=1e-3,
        help="Akseptabelt avvik mellom Python-pipeline og C# ved validering (default: 1e-3)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ckpt = _load_checkpoint(ckpt_path, device)

    train_report_path: Path | None = args.train_report
    if train_report_path is not None and not train_report_path.is_absolute():
        train_report_path = (PROJECT_ROOT / train_report_path).resolve()
    if train_report_path is None:
        candidate = ckpt_path.parent / "train_report.json"
        if candidate.is_file():
            train_report_path = candidate
    train_report = _load_json_if_exists(train_report_path)

    train_cfg: dict[str, Any] | None = None
    train_cfg_path = _resolve_path_from_project(ckpt.get("train_config_path"))
    if train_cfg_path is not None and train_cfg_path.is_file():
        try:
            loaded = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                train_cfg = loaded
        except (OSError, yaml.YAMLError):
            train_cfg = None

    training_block = _resolve_training_for_manifest(
        ckpt=ckpt,
        train_report=train_report,
        train_cfg=train_cfg,
    )
    if args.dataset is not None:
        training_block["dataset"] = args.dataset
    if args.train_samples is not None:
        training_block["samples"] = args.train_samples

    # Inkluder test-metrikkene fra fold_result.json hvis de finnes.
    # Disse er langt mer informative enn val_metrics fra checkpoint.
    fold_result_path = ckpt_path.parent / "fold_result.json"
    if fold_result_path.is_file():
        fold_result = json.loads(fold_result_path.read_text())
        te = fold_result.get("test_eval", {})
        roi = te.get("roi_metrics", {})
        at_half = roi.get("metrics_at_threshold_0.5", {})
        if roi:
            training_block["test_metrics"] = {
                "split": "test",
                "test_patients": fold_result.get("test_patients"),
                "roi_auc_roc": roi.get("auc_roc"),
                "roi_avg_precision": roi.get("avg_precision"),
                "roi_accuracy": at_half.get("accuracy"),
                "roi_f1": at_half.get("f1"),
                "roi_precision": at_half.get("precision"),
                "roi_recall": at_half.get("recall"),
                "optimal_threshold": roi.get("optimal_threshold_youden"),
            }

    cfg_path = args.model_config
    if cfg_path is None:
        raw = ckpt.get("model_config_path")
        if raw is None:
            raise ValueError("Provide --model-config or store 'model_config_path' in checkpoint")
        cfg_path = Path(raw)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()

    pl_path = args.pipeline_config
    if not pl_path.is_absolute():
        pl_path = (PROJECT_ROOT / pl_path).resolve()
    if not pl_path.is_file():
        raise FileNotFoundError(f"Pipeline-config finnes ikke: {pl_path}")

    pipeline = _load_pipeline_dict(pl_path)
    auto_method, pca_path_auto = spectral_from_pipeline(pipeline)
    reducer_method = args.reducer_method if args.reducer_method is not None else auto_method
    reducer_method = str(reducer_method).lower().strip()
    embedded_in_onnx = reducer_method == "pca"

    sk_pca = None
    if embedded_in_onnx:
        pca_path = pca_path_auto
        if pca_path is None:
            # Fallback: les pca_model direkte fra YAML uavhengig av aktiv reducer
            raw_pca = (pipeline.get("spectral_reduction") or {}).get("pca_model")
            if raw_pca:
                pca_path = (PROJECT_ROOT / str(raw_pca)).resolve()
        if pca_path is None or not pca_path.is_file():
            raise FileNotFoundError(
                f"PCA-eksport krever pipeline.spectral_reduction.pca_model (fant ikke {pca_path})"
            )
        sk_pca = joblib.load(pca_path)
        if int(sk_pca.n_components_) != int(args.spectral_bands):
            raise ValueError(
                f"PCA n_components ({sk_pca.n_components_}) må være lik --spectral-bands ({args.spectral_bands})."
            )

    model_yaml = _load_model_yaml(cfg_path)
    cnn = build_model_from_config(cfg_path)
    cnn.load_state_dict(ckpt["model_state_dict"], strict=True)
    cnn.eval()

    h, w = args.patch_h, args.patch_w

    if embedded_in_onnx:
        assert sk_pca is not None
        pca_torch = TorchSpectralPCA(sk_pca)
        export_model: nn.Module = PCAThenCNN(pca_torch, cnn)
        n_raw = pca_torch.n_in
        dummy = torch.randn(1, 1, n_raw, h, w, device=device, dtype=torch.float32)
        input_spec_bands = n_raw
        reducer_in, reducer_out = n_raw, int(args.spectral_bands)
    else:
        export_model = cnn
        c = int(args.spectral_bands)
        dummy = torch.randn(1, 1, c, h, w, device=device, dtype=torch.float32)
        input_spec_bands = c
        raw = args.raw_bands if args.raw_bands is not None else c
        reducer_in, reducer_out = raw, c

    export_model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / args.onnx_name

    torch.onnx.export(
        export_model,
        dummy,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    base: dict[str, Any] | None = None
    if args.manifest_template is not None:
        tpl_path = args.manifest_template.resolve()
        if not tpl_path.is_file():
            raise FileNotFoundError(f"Manifest-mal finnes ikke: {tpl_path}")
        base = json.loads(tpl_path.read_text(encoding="utf-8"))

    preprocessing_block = preprocessing_from_pipeline_yaml(pl_path)

    # Synkroniser preprocessing-blokken med den faktiske reducer_method.
    # Ved PCA brukes "none" fordi PCA er bakt inn i ONNX-grafen — C# skal ikke
    # gjøre noe spektralt reduksjonssteg selv (det håndteres av spectral_reducer-blokken).
    effective_reducer = "none" if embedded_in_onnx else reducer_method
    _sync_preprocessing_reducer(preprocessing_block, effective_reducer, args.spectral_bands)

    # Inkluder min_tissue_ratio fra treningskonfig slik at C# bruker identisk
    # patch-filtrering som Python. Fallback til pipeline.yaml hvis nøkkelen mangler.
    _min_tissue: float | None = None
    if train_cfg is not None:
        _min_tissue = train_cfg.get("data", {}).get("min_tissue_ratio")
    if _min_tissue is None:
        _min_tissue = (pipeline.get("patching") or {}).get("min_tissue_ratio")
    preprocessing_block["params"]["patch_min_tissue_ratio"] = float(_min_tissue) if _min_tissue is not None else 0.0

    manifest = _build_gui_manifest(
        base=base,
        preprocessing=preprocessing_block,
        model=export_model,
        layers_model=cnn if embedded_in_onnx else None,
        model_yaml=model_yaml,
        cfg_path=cfg_path,
        ckpt=ckpt,
        input_spec_bands=input_spec_bands,
        patch_h=h,
        patch_w=w,
        onnx_name=args.onnx_name,
        description=args.description,
        training=training_block,
        reducer_method=reducer_method,
        embedded_reducer_in_onnx=embedded_in_onnx,
        reducer_input_bands=reducer_in,
        reducer_output_bands=reducer_out,
        validation_tolerance=args.validation_tolerance,
    )

    if args.validation_roi_dir is not None:
        roi_dir = args.validation_roi_dir.resolve()
        print(f"[validation] Slicing ROI: {roi_dir}")
        roi_meta = _build_validation_roi(
            roi_dir=roi_dir,
            patch_y=args.validation_patch_y,
            patch_x=args.validation_patch_x,
            patch_h=h,
            patch_w=w,
            out_dir=args.out_dir,
        )
        expected = _compute_expected_output(
            raw_cube=roi_meta["raw_cube"],
            dark_cube=roi_meta["dark_cube"],
            white_cube=roi_meta["white_cube"],
            patch_y=roi_meta["patch_y"],
            patch_x=roi_meta["patch_x"],
            patch_h=h,
            patch_w=w,
            model=cnn,
            reducer_method=reducer_method,
            spectral_bands=args.spectral_bands,
            pipeline=pipeline,
            sk_pca=sk_pca,
        )
        manifest["validation"] = {
            "status": "ready",
            "tolerance": args.validation_tolerance,
            "roi_dir": "roi_validation",
            "patch_coords": {
                "y": roi_meta["patch_y"],
                "x": roi_meta["patch_x"],
                "h": h,
                "w": w,
            },
            "expected_output": expected,
        }
        print(
            f"[validation] Wrote roi_validation/ — predicted_class={expected['predicted_class']} "
            f"softmax={expected['softmax']}"
        )

    out_manifest = args.out_dir / "manifest.json"
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote ONNX: {onnx_path}")
    print(f"Wrote manifest: {out_manifest}")
    print(
        f"[export] reducer={reducer_method} embedded_in_onnx={embedded_in_onnx} "
        f"input (N,1,D,H,W) = (1, 1, {input_spec_bands}, {h}, {w})"
    )


if __name__ == "__main__":
    main()
