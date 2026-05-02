"""Tell trenbare parametre for modeller definert i configs/models/*.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model_from_config


def _count_trainable(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List antall parametre per modell-YAML (default: alle under configs/models/)."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        type=str,
        help="Valgfrie stier til modell-YAML (relativt til prosjektrot eller absolutt).",
    )
    parser.add_argument(
        "--sort",
        choices=("name", "params"),
        default="params",
        help="Sorter etter filnavn eller synkende antall trenbare parametre (default: params).",
    )
    args = parser.parse_args()

    if args.configs:
        paths = [Path(p) for p in args.configs]
    else:
        models_dir = PROJECT_ROOT / "configs" / "models"
        paths = sorted(models_dir.glob("*.yaml"))

    if not paths:
        print("Ingen YAML-filer funnet.", file=sys.stderr)
        return 1

    rows: list[tuple[str, int, int, str | None]] = []
    for raw in paths:
        p = raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
        if not p.is_file():
            rows.append((str(raw), -1, -1, f"mangler: {p}"))
            continue
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        try:
            model = build_model_from_config(p)
            tot, tr = _count_trainable(model)
            rows.append((str(rel), tot, tr, None))
        except Exception as e:  # noqa: BLE001 — vil vise feil per modell
            rows.append((str(rel), -1, -1, str(e)))

    if args.sort == "name":
        rows.sort(key=lambda x: x[0].lower())
    else:
        rows.sort(key=lambda x: x[2], reverse=True)

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'config':<{w}}  {'total':>12}  {'trainable':>12}")
    print("-" * (w + 12 + 12 + 6))
    for name, tot, tr, err in rows:
        if err:
            print(f"{name:<{w}}  {'—':>12}  {'—':>12}  [{err}]")
        else:
            print(f"{name:<{w}}  {tot:12,}  {tr:12,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
