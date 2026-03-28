"""
Offentlig grensesnitt for inferanse (nivå 1).

GUI kaller dette scriptet med --input og --output-dir.
Pipeline: kalibrering → clipping → avg3 → PCA16 → masking → patchifisering.
Skriver primært prediction.json (patches med score 0–1, spatial, tissue_mask, run).
Valgfritt (config): heatmap.png/.npy og reduced_cube.npy for debugging – GUI kan bygge overlay fra JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless for PNG export
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.inference.backend.dummy_backend import predict_dummy
from src.inference.backend.pytorch_backend import predict_pytorch
from src.inference.heatmap import build_heatmap
from src.inference.pipeline import preprocess_single_roi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kjør preprocessing + inferanse på én HSI-ROI."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Sti til ROI-mappe eller raw.hdr.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Mappe der prediction.json skrives (og ev. valgfrie binærfiler).")
    parser.add_argument("--output", type=str, default=None,
                        help="(Alternativ) Sti til prediction.json – output-dir blir parent.")
    parser.add_argument("--config", type=str,
                        default="configs/inference/pytorch.yaml",
                        help="Sti til inference-config.")
    return parser.parse_args()


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """Returner (output_dir, prediction_file) eller None ved feil."""
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        return output_dir, output_dir / "prediction.json"
    if args.output:
        prediction_file = Path(args.output).resolve()
        return prediction_file.parent, prediction_file
    print("[run_inference] ERROR: Mangler --output-dir eller --output", file=sys.stderr)
    return None


def _resolve_config_path(args: argparse.Namespace) -> Path | None:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config
    return config_path if config_path.exists() else None


def _load_model_arch_info(
    project_root: Path,
    model_config_path: str | None,
    cfg_class_names: list[str] | None,
) -> tuple[int, list[str]]:
    """num_classes og class_names for JSON (binær default: normal / anomaly)."""
    default_names = ["normal", "anomaly"]
    if cfg_class_names and len(cfg_class_names) >= 1:
        names = [str(x) for x in cfg_class_names]
        return len(names), names
    if not model_config_path:
        return 2, default_names
    p = Path(model_config_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        return 2, default_names
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    arch = y.get("architecture", {})
    nc = int(arch.get("num_classes", 2))
    names = y.get("class_names")
    if isinstance(names, list) and len(names) == nc:
        return nc, [str(x) for x in names]
    if nc == 2:
        return 2, default_names
    return nc, [f"class_{i}" for i in range(nc)]


def _score_semantics(backend_name: str) -> dict[str, object]:
    if backend_name == "pytorch":
        return {
            "score_type": "softmax_probability",
            "description": (
                "Verdien er P(klasse=positiv | patch) etter softmax på modellens logits. "
                "For binær modell er indeks 1 'anomaly' (se model_info.class_names)."
            ),
            "applies_to_class_index": 1,
        }
    if backend_name == "dummy":
        return {
            "score_type": "heuristic_unit_interval",
            "description": "Syntetisk score for testing (ikke softmax fra trent klassifikator).",
            "applies_to_class_index": None,
        }
    return {
        "score_type": "unknown",
        "description": "",
        "applies_to_class_index": None,
    }


# ---------------------------------------------------------------------------
# Inferanse
# ---------------------------------------------------------------------------

def _run_model(patches: list, coords: list, cfg: dict, project_root: Path) -> tuple[np.ndarray, str]:
    """Kjør modell og returner (scores, backend_name)."""
    model_cfg = cfg.get("model", {})
    backend_name = str(model_cfg.get("backend", "dummy"))

    patches_arr = (
        np.stack(patches, axis=0)
        if patches
        else np.zeros((0, 64, 64, 16), dtype=np.float32)
    )

    if backend_name == "dummy" and len(patches) > 0:
        scores = predict_dummy(
            patches_arr, coords, seed=int(cfg.get("seed", 42))
        )
    elif backend_name == "pytorch" and len(patches) > 0:
        checkpoint_path = model_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("model.checkpoint_path required when backend=pytorch")
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            ckpt = (project_root / ckpt).resolve()
        scores = predict_pytorch(
            patches_arr,
            checkpoint_path=ckpt,
            model_config_path=model_cfg.get("model_config_path"),
            project_root=project_root,
            batch_size=int(model_cfg.get("batch_size", 32)),
        )
    else:
        scores = np.zeros(len(patches), dtype=np.float32)

    return scores, backend_name


def _sync_accelerator_if_needed() -> None:
    """Ensure queued GPU work is finished so wall-clock timing includes accelerator time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _build_heatmap(coords: list, scores: np.ndarray, reduced_cube: np.ndarray, metadata: dict) -> np.ndarray:
    h, w, _ = reduced_cube.shape
    patch_h = int(metadata.get("patch_h", 64))
    patch_w = int(metadata.get("patch_w", 64))
    return build_heatmap(coords, scores, patch_h, patch_w, h, w)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save_heatmap_png(heatmap: np.ndarray, path: Path, width: int, height: int) -> None:
    """Lagre heatmap som PNG (lav=blå, høy=rød)."""
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)
    ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1, aspect="equal")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)


def _write_error(output_dir: Path, error: str, prediction_file: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2,
        "status": "error",
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = prediction_file or (output_dir / "prediction.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[run_inference] ERROR: {error}", file=sys.stderr)


def _save_outputs(
    output_dir: Path,
    prediction: dict,
    heatmap: np.ndarray | None,
    reduced_cube: np.ndarray | None,
    *,
    write_heatmap_assets: bool,
    write_reduced_cube: bool,
) -> None:
    """Skriv prediction.json; binærfiler kun hvis flagg er True."""
    with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    if write_reduced_cube and reduced_cube is not None:
        np.save(output_dir / "reduced_cube.npy", reduced_cube)

    if write_heatmap_assets and heatmap is not None and reduced_cube is not None:
        np.save(output_dir / "heatmap.npy", heatmap)
        h, w, _ = reduced_cube.shape
        _save_heatmap_png(heatmap, output_dir / "heatmap.png", w, h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    paths = _resolve_output_paths(args)
    if paths is None:
        return 1
    output_dir, prediction_file = paths

    config_path = _resolve_config_path(args)
    if config_path is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_error(output_dir, f"Config not found: {args.config}", prediction_file)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    # Preprocessing
    t_cpu_preprocess_0 = time.process_time()
    t_wall_preprocess_0 = time.perf_counter()
    try:
        patches, coords, reduced_cube, metadata = preprocess_single_roi(
            input_path=args.input,
            config_path=config_path,
            project_root=PROJECT_ROOT,
        )
    except FileNotFoundError as e:
        _write_error(output_dir, str(e), prediction_file)
        return 1
    except Exception as e:
        _write_error(output_dir, str(e), prediction_file)
        return 1
    preprocess_wall_ms = int((time.perf_counter() - t_wall_preprocess_0) * 1000)
    preprocess_cpu_ms = int((time.process_time() - t_cpu_preprocess_0) * 1000)

    # Inferanse
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _sync_accelerator_if_needed()
    t_cpu_model_0 = time.process_time()
    t_wall_model_0 = time.perf_counter()
    scores, backend_name = _run_model(patches, coords, cfg, PROJECT_ROOT)
    _sync_accelerator_if_needed()
    model_wall_ms = int((time.perf_counter() - t_wall_model_0) * 1000)
    model_cpu_ms = int((time.process_time() - t_cpu_model_0) * 1000)

    write_heatmap_assets = bool(cfg.get("write_heatmap_assets", False))
    write_reduced_cube = bool(cfg.get("write_reduced_cube", False))
    heatmap: np.ndarray | None = None
    if write_heatmap_assets:
        heatmap = _build_heatmap(coords, scores, reduced_cube, metadata)

    ps = metadata["patch_stats"]
    if ps["evaluated"] != len(coords) or len(patches) != len(coords):
        raise RuntimeError(
            f"Internal patch count mismatch: patch_stats.evaluated={ps['evaluated']}, "
            f"len(coords)={len(coords)}, len(patches)={len(patches)}"
        )

    model_cfg = cfg.get("model", {})
    decision_cfg = cfg.get("decision", {})
    threshold = float(decision_cfg.get("anomaly_threshold", 0.5))

    num_classes, class_names = _load_model_arch_info(
        PROJECT_ROOT,
        model_cfg.get("model_config_path"),
        model_cfg.get("class_names"),
    )
    positive_class = str(decision_cfg.get("positive_class", "anomaly"))
    if positive_class not in class_names and len(class_names) >= 2:
        positive_class = class_names[1]

    ckpt_raw = model_cfg.get("checkpoint_path")
    ckpt_path = Path(ckpt_raw) if ckpt_raw else None
    if ckpt_path and not ckpt_path.is_absolute():
        ckpt_resolved = (PROJECT_ROOT / ckpt_path).resolve()
    elif ckpt_path:
        ckpt_resolved = ckpt_path.resolve()
    else:
        ckpt_resolved = None

    predictions: list[dict] = []
    for i, ((y, x), s) in enumerate(zip(coords, scores)):
        s = float(s)
        is_anom = s >= threshold
        label = positive_class if is_anom else (class_names[0] if class_names else "normal")
        if num_classes == 2 and len(class_names) >= 2:
            probs = {class_names[0]: float(1.0 - s), class_names[1]: float(s)}
        else:
            probs = {positive_class: float(s)}
        predictions.append(
            {
                "id": i,
                "y": int(y),
                "x": int(x),
                "score": s,
                "probabilities": probs,
                "label": label,
            }
        )

    anomaly_count = int(np.sum(scores >= threshold))
    duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    patch_stats = metadata["patch_stats"]
    cube_shape = metadata["cube_shape"]
    n_bands = int(cube_shape[2]) if len(cube_shape) >= 3 else 0

    prediction = {
        "schema_version": 2,
        "status": "ok",
        "input": {
            "path": args.input,
            "timestamp": start.isoformat(),
        },
        "model_info": {
            "name": str(model_cfg.get("name", "dummy_3dcnn")),
            "backend": backend_name,
            "checkpoint_path": str(ckpt_raw) if ckpt_raw else None,
            "checkpoint_file": ckpt_resolved.name if ckpt_resolved and ckpt_resolved.exists() else None,
            "model_config_path": model_cfg.get("model_config_path"),
            "num_classes": num_classes,
            "class_names": class_names,
        },
        "decision": {
            "positive_class": positive_class,
            "anomaly_threshold": threshold,
            **_score_semantics(backend_name),
        },
        "spatial": {
            "coordinate_space": "reduced_cube",
            "coordinate_space_description": (
                "Koordinater (y, x) er rad/kolonne i den spektralt reduserte kuben "
                "(etter kalibrering, clip, avg3, tissue-mask på mellomkube, PCA/AE/wavelet). "
                "Ikke rå HDR uten videre."
            ),
            "axes_order": ["y_lines", "x_samples", "spectral"],
            "cube_shape": cube_shape,
            "patch_h": metadata["patch_h"],
            "patch_w": metadata["patch_w"],
            "stride_h": metadata["stride_h"],
            "stride_w": metadata["stride_w"],
            "patch_anchor": "top_left",
            "origin": "top_left",
        },
        "preprocessing": {
            "pipeline_config": metadata.get("pipeline_config_relative"),
            "steps": metadata.get("pipeline_steps", {}),
            "spectral_reducer": metadata.get("spectral_reducer"),
            "num_spectral_bands": n_bands,
            "min_tissue_ratio_patch": metadata.get("min_tissue_ratio_patch"),
            "compression_input_bytes": metadata.get("compression_input_bytes"),
            "compression_output_bytes": metadata.get("compression_output_bytes"),
            "file_compression_ratio": metadata.get("file_compression_ratio"),
            "compression_input_description": metadata.get("compression_input_description"),
            "compression_output_description": metadata.get("compression_output_description"),
        },
        "tissue_mask": metadata["tissue_mask"],
        "patch_stats": {
            "total_possible": patch_stats["total_possible"],
            "evaluated": patch_stats["evaluated"],
            "filtered_by_tissue": patch_stats["filtered_by_tissue"],
            "description": (
                "total_possible: gyldige grid-posisjoner; filtered_by_tissue: hoppet pga. "
                "min_tissue_ratio_patch i patch-vindu; evaluated: patches sendt til modell."
            ),
        },
        "predictions": predictions,
        "summary": {
            "anomaly_ratio": round(anomaly_count / max(1, len(predictions)), 4),
            "message": (
                f"Inference OK. {patch_stats['evaluated']} patches evaluert "
                f"({patch_stats['filtered_by_tissue']} filtrert av vev-krav), "
                f"{anomaly_count} med score >= {threshold}."
            ),
        },
        "run": {
            "duration_ms": duration_ms,
            "preprocess_wall_ms": preprocess_wall_ms,
            "model_wall_ms": model_wall_ms,
            "preprocess_cpu_ms": preprocess_cpu_ms,
            "model_cpu_ms": model_cpu_ms,
            "timing_note": (
                "Wall times use perf_counter (includes I/O wait). preprocess_cpu_ms / "
                "model_cpu_ms are process CPU time deltas (time.process_time); they do not "
                "map to hardware cycle counts. With CUDA, model_wall_ms uses "
                "cuda.synchronize() / mps.synchronize() so accelerator work is included. "
                "For true CPU cycles or GPU "
                "kernels, use OS/profiler tools (e.g. Intel VTune, NVIDIA Nsight)."
            ),
        },
    }

    _save_outputs(
        output_dir,
        prediction,
        heatmap,
        reduced_cube,
        write_heatmap_assets=write_heatmap_assets,
        write_reduced_cube=write_reduced_cube,
    )

    extra = []
    if write_heatmap_assets:
        extra.append("heatmap.png")
    if write_reduced_cube:
        extra.append("reduced_cube.npy")
    suffix = f" + {', '.join(extra)}" if extra else ""
    cr = metadata.get("file_compression_ratio")
    cr_s = f", file compression≈{cr:.2f}×" if isinstance(cr, (int, float)) else ""
    print(
        f"[run_inference] OK – {metadata['num_patches']} patches{cr_s}, "
        f"preprocess {preprocess_wall_ms} ms, model {model_wall_ms} ms → {output_dir}{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
