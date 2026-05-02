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
    """Set up root logger with a rotating file handler and optional console handler.

    Idempotent: if the root logger already has handlers this call is a no-op,
    so it is safe to call multiple times from scripts and modules.

    Args:
        log_dir: Directory where log files are written (created if absent).
        run_name: Base name for the log file ({run_name}.log).
        level: Logging level applied to root logger and all handlers.
        console: If True, also attach a StreamHandler to stdout.
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
    """Return a named logger.

    Every module should call ``get_logger(__name__)`` at module level instead of
    using ``print()``.  ``configure_logging()`` must be called once (in the
    entry-point script) before any log messages are emitted.
    """
    return logging.getLogger(name)
