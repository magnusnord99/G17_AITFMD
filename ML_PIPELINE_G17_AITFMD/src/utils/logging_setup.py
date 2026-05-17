"""Logging configuration for the ML pipeline."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_dir: str | Path,
    run_name: str,
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Sett opp root-logger med roterende filhåndterer og valgfri konsollhåndterer.

    Idempotent: trygt å kalle flere ganger — no-op om root-logger allerede har handlere.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    log_path = Path(log_dir) / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root.addHandler(sh)


def get_logger(name: str) -> logging.Logger:
    """Returner navngitt logger. Kall configure_logging() én gang fra entry-point før bruk."""
    return logging.getLogger(name)
