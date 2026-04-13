"""I/O utilities: config loading, JSON helpers, directory management."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | os.PathLike) -> dict:
    """
    Load a YAML configuration file and return it as a nested dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Nested dictionary representing the configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded config from %s", path)
    return config


def save_json(data: Any, path: str | os.PathLike, indent: int = 2) -> None:
    """
    Serialize *data* to JSON and write it to *path*.

    The parent directory is created automatically if it does not exist.

    Args:
        data:   JSON-serialisable Python object.
        path:   Destination file path.
        indent: JSON indentation level (default 2).
    """
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug("Saved JSON to %s", path)


def load_json(path: str | os.PathLike) -> Any:
    """
    Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug("Loaded JSON from %s", path)
    return data


def ensure_dir(path: str | os.PathLike) -> Path:
    """
    Create *path* (and all intermediate directories) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
