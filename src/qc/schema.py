"""
qc/schema.py — Minimal schema and integrity checks for the WildfireRisk-EU DuckDB.

Checks performed:
  1. Required tables exist.
  2. Required columns are present in each table.
  3. Critical score columns have no NULLs.

Designed to be called from result_table.py (T12) before writing outputs.
Exits with a non-zero status and a clear message on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Tables that must exist after the full pipeline has run
REQUIRED_TABLES = [
    "buildings",
    "features_terrain",
    "features_vegetation",
    "features_fire_weather",
    "features_fire_history",
    "risk_scores",
]

# Minimum columns required per table (not exhaustive — spot-check only)
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "buildings": ["building_id", "centroid_lat", "centroid_lon", "area_m2"],
    "risk_scores": [
        "building_id",
        "composite_score",
        "risk_class",
        "score_terrain",
        "score_vegetation",
        "score_fire_weather",
        "score_fire_history",
    ],
    "features_terrain": ["building_id", "elevation_m", "slope_deg"],
    "features_fire_history": ["building_id", "dist_to_nearest_fire_m"],
}

# Columns that must contain zero NULLs in risk_scores
NON_NULL_SCORE_COLS = ["composite_score", "risk_class"]


def check_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return list of required tables that are missing."""
    existing = {
        row[0] for row in con.execute("SHOW TABLES").fetchall()
    }
    return [t for t in REQUIRED_TABLES if t not in existing]


def check_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return list of 'table.column' pairs that are missing."""
    missing = []
    for table, cols in REQUIRED_COLUMNS.items():
        try:
            existing_cols = {
                row[0]
                for row in con.execute(f"DESCRIBE {table}").fetchall()
            }
        except Exception:
            # Table missing — already caught by check_tables
            continue
        for col in cols:
            if col not in existing_cols:
                missing.append(f"{table}.{col}")
    return missing


def check_nulls(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return list of 'table.column: N nulls' strings where nulls > 0."""
    violations = []
    for col in NON_NULL_SCORE_COLS:
        try:
            n = con.execute(
                f"SELECT COUNT(*) FROM risk_scores WHERE {col} IS NULL"
            ).fetchone()[0]
        except Exception as exc:
            violations.append(f"risk_scores.{col}: query failed ({exc})")
            continue
        if n > 0:
            violations.append(f"risk_scores.{col}: {n:,} NULLs")
    return violations


def run_all(db_path: Path, exit_on_failure: bool = True) -> bool:
    """
    Run all QC checks against the DuckDB at *db_path*.

    Returns True if all checks pass.
    If *exit_on_failure* is True (default), calls sys.exit(1) on failure.
    """
    if not db_path.exists():
        msg = f"[qc] ERROR: database not found: {db_path}"
        if exit_on_failure:
            sys.exit(msg)
        print(msg)
        return False

    con = duckdb.connect(str(db_path), read_only=True)

    failures: list[str] = []

    missing_tables = check_tables(con)
    if missing_tables:
        failures.append(f"Missing tables: {', '.join(missing_tables)}")

    missing_cols = check_columns(con)
    if missing_cols:
        failures.append(f"Missing columns: {', '.join(missing_cols)}")

    null_violations = check_nulls(con)
    if null_violations:
        failures.append(f"NULL violations: {'; '.join(null_violations)}")

    con.close()

    if failures:
        msg = "[qc] SCHEMA CHECK FAILED\n" + "\n".join(f"  - {f}" for f in failures)
        if exit_on_failure:
            sys.exit(msg)
        print(msg)
        return False

    print(f"[qc] Schema checks passed ({len(REQUIRED_TABLES)} tables, "
          f"{sum(len(v) for v in REQUIRED_COLUMNS.values())} columns, "
          f"{len(NON_NULL_SCORE_COLS)} null checks)")
    return True
