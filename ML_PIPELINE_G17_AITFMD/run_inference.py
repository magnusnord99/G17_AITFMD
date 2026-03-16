"""
Offentlig grensesnitt for inferanse (nivå 1).

GUI kaller dette scriptet med --input og --output-dir.
Pipeline: kalibrering → clipping → avg3 → PCA16 → masking → patchifisering.
Skriver prediction.json og metadata.json til output-dir.
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
from src.inference.heatmap import build_heatmap
from src.inference.pipeline import preprocess_single_roi


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kjør preprocessing + inferanse på én HSI-ROI."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Sti til ROI-mappe eller raw.hdr.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Mappe der prediction.json og metadata.json skrives.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(Alternativ) Sti til prediction.json – output-dir blir parent.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/default.yaml",
        help="Sti til inference-config.",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        prediction_file = output_dir / "prediction.json"
    elif args.output:
        prediction_file = Path(args.output).resolve()
        output_dir = prediction_file.parent
    else:
        print("[run_inference] ERROR: Mangler --output-dir eller --output", file=sys.stderr)
        return 1
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        config_path = PROJECT_ROOT / args.config

    if not config_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_error(output_dir, f"Config not found: {config_path}", prediction_file)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

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

    # Run dummy model (for GUI testing until real 3D CNN is trained)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg.get("model", {})
    backend_name = str(model_cfg.get("backend", "dummy"))

    patches_arr = np.stack(patches, axis=0) if patches else np.zeros((0, 64, 64, 16), dtype=np.float32)
    if backend_name == "dummy" and len(patches) > 0:
        scores = predict_dummy(patches_arr, coords, seed=int(cfg.get("seed", 42)))
    else:
        scores = np.zeros(len(patches), dtype=np.float32)

    # Build heatmap for overlay
    h, w, _ = reduced_cube.shape
    patch_h = int(cfg.get("preprocessing", {}).get("patching", {}).get("patch_h", 64))
    patch_w = int(cfg.get("preprocessing", {}).get("patching", {}).get("patch_w", 64))
    heatmap = build_heatmap(coords, scores, patch_h, patch_w, h, w)

    # Per-patch predictions for JSON
    predictions = [
        {"y": int(y), "x": int(x), "score": float(s), "label": "anomaly" if s > 0.5 else "normal"}
        for (y, x), s in zip(coords, scores)
    ]

    duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    anomaly_count = int(np.sum(scores > 0.5))

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

    metadata_path = output_dir / "metadata.json"
    heatmap_path = output_dir / "heatmap.npy"

    with prediction_file.open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.save(heatmap_path, heatmap)

    # Save heatmap as PNG for GUI overlay (colormap: lav=blå, høy=rød)
    heatmap_png = output_dir / "heatmap.png"
    fig, ax = plt.subplots(figsize=(w / 80, h / 80), dpi=80)
    ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1, aspect="equal")
    ax.axis("off")
    fig.savefig(heatmap_png, bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)

    reduced_path = output_dir / "reduced_cube.npy"
    np.save(reduced_path, reduced_cube)

    print(f"[run_inference] OK – {metadata['num_patches']} patches, heatmap skrevet til {output_dir}")
    return 0


def _write_error(output_dir: Path, error: str, prediction_file: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    err_result = {
        "status": "error",
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = prediction_file or (output_dir / "prediction.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(err_result, f, indent=2, ensure_ascii=False)
    print(f"[run_inference] ERROR: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
