"""
config.py — Merged config loader for the WildfireRisk-EU pipeline.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    bbox = cfg["pipeline"]["aoi"]["bbox"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Config files loaded in this order; later files can override earlier ones.
_CONFIG_FILES = [
    "config/pipeline.yaml",
    "config/data_sources.yaml",
    "config/features.yaml",
    "config/scoring.yaml",
    "config/validation.yaml",
]


def _find_project_root() -> Path:
    """Walk up from this file's location until a pyproject.toml is found."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml found in parent directories). "
        "Run the pipeline from the project root directory."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(extra_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load and merge all pipeline config files into a single dict.

    The returned dict has one top-level key per config file:
        cfg["pipeline"]    — pipeline.yaml
        cfg["sources"]     — data_sources.yaml
        cfg["features"]    — features.yaml
        cfg["scoring"]     — scoring.yaml
        cfg["validation"]  — validation.yaml

    Args:
        extra_overrides: Optional dict merged over the loaded config at the
            top level. Useful for per-run CLI overrides.

    Returns:
        Merged configuration dict.
    """
    root = _find_project_root()

    # Map file stem → config key
    key_map = {
        "pipeline": "pipeline",
        "data_sources": "sources",
        "features": "features",
        "scoring": "scoring",
        "validation": "validation",
    }

    cfg: dict[str, Any] = {}
    for rel_path in _CONFIG_FILES:
        full_path = root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")
        stem = full_path.stem
        key = key_map.get(stem, stem)
        cfg[key] = _load_yaml(full_path)

    if extra_overrides:
        cfg.update(extra_overrides)

    return cfg


def get_project_root() -> Path:
    """Return the absolute project root path."""
    return _find_project_root()


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    return _find_project_root() / relative_path


def get_bbox(cfg: dict[str, Any]) -> list[float]:
    """Return AOI bbox as [west, south, east, north]."""
    return cfg["pipeline"]["aoi"]["bbox"]


def get_crs(cfg: dict[str, Any], key: str = "crs_working") -> str:
    """Return CRS string from pipeline config. key: 'crs_working' or 'crs_output'."""
    return cfg["pipeline"]["aoi"][key]


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully.")
    print(f"  Project : {cfg['pipeline']['project']['name']}")
    print(f"  AOI bbox: {cfg['pipeline']['aoi']['bbox']}")
    print(f"  CRS     : {cfg['pipeline']['aoi']['crs_working']}")
    print(f"  DB path : {cfg['pipeline']['paths']['db']}")
