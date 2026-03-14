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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    duration_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)

    prediction = {
        "status": "ok",
        "input_path": args.input,
        "timestamp": start.isoformat(),
        "summary": {
            "num_patches": metadata["num_patches"],
            "cube_shape": metadata["cube_shape"],
            "message": f"Preprocessing OK. {metadata['num_patches']} patches ekstrahert.",
        },
        "patch_coords_sample": coords[:5] if coords else [],
    }

    meta = {
        "model": metadata.get("spectral_reducer", "ae"),
        "backend": "preprocessing",
        "duration_ms": duration_ms,
        "num_patches": metadata["num_patches"],
    }

    metadata_path = output_dir / "metadata.json"

    with prediction_file.open("w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    reduced_path = output_dir / "reduced_cube.npy"
    np.save(reduced_path, reduced_cube)

    print(f"[run_inference] OK – {metadata['num_patches']} patches, skrev til {output_dir}")
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
