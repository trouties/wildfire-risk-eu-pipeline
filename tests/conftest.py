"""
conftest.py — Shared pytest fixtures for WildfireRisk-EU test suite.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg(project_root):
    """Load merged pipeline config once per test session."""
    from src.utils.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def sample_data_dir(project_root) -> Path:
    return project_root / "data" / "sample"


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary DuckDB file (deleted after each test)."""
    return tmp_path / "test_wildfire.duckdb"
