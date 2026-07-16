"""Utility functions module."""

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name.
        level: Log level.
        log_file: Log file path; if None, logs to console only.
        fmt: Log format string.

    Returns:
        Configured logger.

    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.

    Returns:
        Path object of the directory.

    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, **kwargs: Any) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save.
        path: File path.
        kwargs: Extra arguments passed to json.dump.

    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, **kwargs)


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Args:
        path: File path.

    Returns:
        Parsed data.

    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_project_root() -> Path:
    """Return the project root directory (parent of src/).

    Returns:
        Path object of the project root.

    """
    return Path(__file__).resolve().parent.parent
