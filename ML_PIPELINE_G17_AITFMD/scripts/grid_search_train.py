#!/usr/bin/env python3
"""
Kjør grid search over train.yaml-hyperparametre ved å kalle scripts/run_train.py sekvensielt.

Bruk:
  python scripts/grid_search_train.py --grid configs/grid_search/nightly.yaml
  python scripts/grid_search_train.py --grid configs/grid_search/nightly.yaml --dry-run
  python scripts/grid_search_train.py --grid configs/grid_search/nightly.yaml --max-runs 2

Krever at run_train.py fullfører (hyperparams.yaml / hyperparams.json skrives ved suksess).
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _load_grid_spec(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "grid" not in data:
        raise ValueError("Grid file must contain a top-level 'grid:' mapping")
    return data


def _short_tag(overrides: dict[str, Any]) -> str:
    """Kort filnavn-tag fra overrides."""
    bits = []
    for k, v in sorted(overrides.items()):
        key = k.split(".")[-1][:8]
        if isinstance(v, float):
            bits.append(f"{key}{v:.1e}".replace("e+0", "e").replace("e-0", "e-"))
        else:
            bits.append(f"{key}{v}")
    return "_".join(bits)[:120] or "run"


def _reports_dir_from_train_cfg(cfg: dict[str, Any]) -> Path:
    paths_cfg = cfg.get("paths") or {}
    raw = str(paths_cfg.get("reports_dir", "outputs/reports"))
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _read_newest_train_report(reports_dir: Path, after_wall_time: float | None = None) -> dict[str, Any]:
    """Les nyeste train_report_*.json. Hvis after_wall_time er satt, kun filer endret etter det (unngår forrige run)."""
    candidates = list(reports_dir.glob("train_report_*.json"))
    if after_wall_time is not None:
        filtered = [p for p in candidates if p.stat().st_mtime >= after_wall_time - 1.0]
        candidates = filtered if filtered else candidates
    if not candidates:
        return {}
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "report_path": str(newest),
        "best_val_loss": data.get("best_val_loss"),
        "best_epoch": data.get("best_epoch"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search over train.yaml (sequential runs).")
    parser.add_argument(
        "--grid",
        type=str,
        default="configs/grid_search/nightly.yaml",
        help="YAML med base_config + grid (liste per hyperparameter)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skriv kombinasjoner og avslutt")
    parser.add_argument("--max-runs", type=int, default=None, help="Begrens antall kjøringer (test)")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python til å kjøre run_train.py",
    )
    args = parser.parse_args()

    grid_path = Path(args.grid).expanduser()
    if not grid_path.is_absolute():
        grid_path = (PROJECT_ROOT / grid_path).resolve()
    if not grid_path.exists():
        print(f"[grid] ERROR: grid file not found: {grid_path}", flush=True)
        return 1

    spec = _load_grid_spec(grid_path)
    base_rel = spec.get("base_config", "configs/train.yaml")
    base_path = PROJECT_ROOT / base_rel
    if not base_path.exists():
        print(f"[grid] ERROR: base config not found: {base_path}", flush=True)
        return 1

    grid: dict[str, list[Any]] = spec["grid"]
    if not grid:
        print("[grid] ERROR: grid is empty", flush=True)
        return 1

    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combinations = list(itertools.product(*value_lists))
    if args.max_runs is not None:
        combinations = combinations[: max(0, args.max_runs)]

    out_prefix = spec.get("output_prefix", "outputs/grid_search")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = PROJECT_ROOT / out_prefix
    session_dir.mkdir(parents=True, exist_ok=True)
    run_configs_dir = session_dir / f"configs_{session_id}"
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = session_dir / f"summary_{session_id}.csv"

    print(f"[grid] base_config={base_path}", flush=True)
    print(f"[grid] combinations={len(combinations)}", flush=True)
    print(f"[grid] session_dir={session_dir}", flush=True)
    print(f"[grid] summary_csv={summary_csv}", flush=True)

    if args.dry_run:
        for i, combo in enumerate(combinations):
            overrides = dict(zip(keys, combo))
            print(f"  {i+1:3d}  {overrides}", flush=True)
        return 0

    fieldnames = [
        "run_index",
        "exit_code",
        "duration_sec",
        "report_path",
        "best_val_loss",
        "best_epoch",
        "error",
        *keys,
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i, combo in enumerate(combinations):
            overrides = dict(zip(keys, combo))
            tag = _short_tag(overrides)
            cfg_copy = copy.deepcopy(yaml.safe_load(base_path.read_text(encoding="utf-8")))
            for k, v in overrides.items():
                _set_nested(cfg_copy, k, v)

            cfg_name = f"run_{i+1:03d}_{tag}.yaml"
            cfg_path = run_configs_dir / cfg_name
            cfg_path.write_text(
                yaml.safe_dump(cfg_copy, sort_keys=False, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )

            rel_cfg = cfg_path.relative_to(PROJECT_ROOT)
            reports_dir = _reports_dir_from_train_cfg(cfg_copy)

            print(f"\n[grid] ===== run {i+1}/{len(combinations)} {overrides} =====", flush=True)
            print(f"[grid] config -> {cfg_path}", flush=True)
            print(
                "[grid] Live output from run_train below (ingen blokkering — samme som direkte kjøring).\n",
                flush=True,
            )

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            t0 = time.perf_counter()
            t_wall = time.time()
            # Ikke capture_output: da ser du tqdm og epoker fortløpende (ellers tomt i timevis).
            proc = subprocess.run(
                [args.python, "-u", str(PROJECT_ROOT / "scripts/run_train.py"), "--config", str(rel_cfg)],
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            elapsed = time.perf_counter() - t0

            from_disk = _read_newest_train_report(reports_dir, after_wall_time=t_wall)
            bvl = from_disk.get("best_val_loss", "")
            bep = from_disk.get("best_epoch", "")
            rpath = from_disk.get("report_path", "")

            row: dict[str, Any] = {
                "run_index": i + 1,
                "exit_code": proc.returncode,
                "duration_sec": round(elapsed, 1),
                "report_path": rpath,
                "best_val_loss": bvl if bvl != "" else "",
                "best_epoch": bep if bep != "" else "",
                "error": "",
            }
            if proc.returncode != 0:
                row["error"] = "run_train exited non-zero; see terminal output above"

            for k in keys:
                row[k] = overrides[k]

            writer.writerow(row)
            fcsv.flush()

            print(
                f"\n[grid] run {i+1} done exit={proc.returncode} time={elapsed/60.0:.1f} min "
                f"best_val_loss={bvl} best_epoch={bep}",
                flush=True,
            )

    print(f"\n[grid] ALL DONE. Summary: {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
