"""
Offentlig grensesnitt for inferanse (nivå 1).

GUI kaller dette scriptet med --input og --output-dir.
Pipeline: kalibrering → clipping → avg3 → PCA16 → masking → patchifisering.
Skriver prediction.json, metadata.json og heatmap.png til output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless for PNG export
import matplotlib.pyplot as plt
import numpy as np

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
                        help="Mappe der prediction.json og metadata.json skrives.")
    parser.add_argument("--output", type=str, default=None,
                        help="(Alternativ) Sti til prediction.json – output-dir blir parent.")
    parser.add_argument("--config", type=str,
                        default="configs/inference/default.yaml",
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


def _build_heatmap(coords: list, scores: np.ndarray, reduced_cube: np.ndarray, cfg: dict) -> np.ndarray:
    h, w, _ = reduced_cube.shape
    patching = cfg.get("preprocessing", {}).get("patching", {})
    patch_h = int(patching.get("patch_h", 64))
    patch_w = int(patching.get("patch_w", 64))
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
    meta: dict,
    heatmap: np.ndarray,
    reduced_cube: np.ndarray,
) -> None:
    """Skriv alle output-filer til output_dir."""
    with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.save(output_dir / "heatmap.npy", heatmap)
    np.save(output_dir / "reduced_cube.npy", reduced_cube)

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

    # Inferanse
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scores, backend_name = _run_model(patches, coords, cfg, PROJECT_ROOT)
    heatmap = _build_heatmap(coords, scores, reduced_cube, cfg)

    # Bygg resultat-dicts
    model_cfg = cfg.get("model", {})
    predictions = [
        {"y": int(y), "x": int(x), "score": float(s), "label": "anomaly" if s > 0.5 else "normal"}
        for (y, x), s in zip(coords, scores)
    ]
    anomaly_count = int(np.sum(scores > 0.5))
    duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

    prediction = {
        "status": "ok",
        "input_path": args.input,
        "timestamp": start.isoformat(),
        "predictions": predictions,
        "summary": {
            "num_patches": metadata["num_patches"],
            "cube_shape": metadata["cube_shape"],
            "anomaly_ratio": round(anomaly_count / max(1, len(predictions)), 4),
            "message": f"Inference OK. {metadata['num_patches']} patches, {anomaly_count} med høy sannsynlighet.",
        },
        "heatmap_path": "heatmap.png",
        "patch_coords_sample": coords[:5] if coords else [],
    }

    meta = {
        "model": model_cfg.get("name", "dummy_3dcnn"),
        "backend": backend_name,
        "duration_ms": duration_ms,
        "num_patches": metadata["num_patches"],
    }

    _save_outputs(output_dir, prediction, meta, heatmap, reduced_cube)

    print(f"[run_inference] OK – {metadata['num_patches']} patches, heatmap skrevet til {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
