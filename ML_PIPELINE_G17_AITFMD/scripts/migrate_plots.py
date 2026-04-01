#!/usr/bin/env python3
"""Migrer outputs/plots fra flat struktur til per-modell-struktur.

Gammelt: outputs/plots/<model_name>_<run_id>/
Nytt:    outputs/plots/<model_name>/<run_id>/

Kjøres én gang. Hopper over mapper som ikke matcher mønsteret.
Stopper med feil hvis en destinasjon allerede finnes (ikke overskriver).

Bruk:
    python scripts/migrate_plots.py [--dry-run] [--plots-dir outputs/plots]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# run_id er alltid YYYYMMDD_HHMMSS (8 siffer, underscore, 6 siffer)
_RUN_ID_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")


def migrate(plots_dir: Path, dry_run: bool) -> int:
    if not plots_dir.exists():
        print(f"[migrate] ERROR: plots_dir not found: {plots_dir}")
        return 1

    candidates = [d for d in sorted(plots_dir.iterdir()) if d.is_dir()]
    matched: list[tuple[Path, str, str]] = []
    skipped: list[Path] = []

    for d in candidates:
        m = _RUN_ID_PATTERN.match(d.name)
        if m:
            model_name, run_id = m.group(1), m.group(2)
            matched.append((d, model_name, run_id))
        else:
            skipped.append(d)

    print(f"[migrate] plots_dir : {plots_dir}")
    print(f"[migrate] to migrate: {len(matched)}")
    print(f"[migrate] skipping  : {len(skipped)} (ikke-matchende mapper)")
    if skipped:
        for d in skipped:
            print(f"  skip: {d.name}")

    errors = 0
    moved = 0

    for src, model_name, run_id in matched:
        dest_parent = plots_dir / model_name
        dest = dest_parent / run_id

        if dest.exists():
            print(f"  [SKIP] destinasjon finnes allerede: {dest.relative_to(plots_dir)}")
            errors += 1
            continue

        action = "DRY-RUN" if dry_run else "FLYTT"
        print(f"  [{action}] {src.name}  →  {model_name}/{run_id}/")

        if not dry_run:
            dest_parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1

    if dry_run:
        print(f"\n[migrate] Tørrkjøring fullført — ingen filer er flyttet.")
        print(f"[migrate] Kjør uten --dry-run for å utføre migreringen.")
    else:
        print(f"\n[migrate] Ferdig. Flyttet: {moved}  Feil/skippet: {errors}")

    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrer plot-mapper til per-modell-struktur.")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="Sti til plots-mappen (relativ til prosjektrot)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Vis hva som ville blitt gjort, men flytt ingenting",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    if not plots_dir.is_absolute():
        plots_dir = (PROJECT_ROOT / plots_dir).resolve()

    return migrate(plots_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
