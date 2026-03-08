"""
Offentlig grensesnitt for inferanse (nivå 1).

GUI kaller dette scriptet; det skriver en resultatfil som frontend kan lese.
Per nå: kun mock-data for testing av integrasjonen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def _write_mock_result(output_path: Path, input_path: str) -> None:
    """Skriver mock-inferanseresultat til JSON-fil."""
    result = {
        "status": "ok",
        "mock": True,
        "input_path": str(input_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions": [
            {"x": 10, "y": 20, "score": 0.92, "label": "anomaly"},
            {"x": 50, "y": 80, "score": 0.15, "label": "normal"},
            {"x": 120, "y": 45, "score": 0.78, "label": "anomaly"},
        ],
        "summary": {
            "anomaly_ratio": 0.12,
            "total_pixels": 1000,
            "message": "Mock resultat – ekte inferanse ikke implementert ennå.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kjør inferanse på HSI-cube. (Mock: skriver kun testdata.)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Sti til input (f.eks. .hdr-fil). Brukes kun til logging i mock.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Sti til resultatfil (JSON) som frontend leser.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    _write_mock_result(output_path, args.input or "(ingen input)")

    print(f"[run_inference] Mock OK – skrev til {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
