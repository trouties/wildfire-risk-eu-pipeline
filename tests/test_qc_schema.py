"""
tests/test_qc_schema.py
------------------------
Unit tests for src/qc/schema.py — schema and integrity checks.

Uses a temporary DuckDB per test (via tmp_path), no network or credentials.
"""

from __future__ import annotations

import duckdb
import pytest

from src.qc.schema import (
    NON_NULL_SCORE_COLS,
    REQUIRED_TABLES,
    check_columns,
    check_nulls,
    check_tables,
    run_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_minimal_db(db_path, include_tables=None, nulls_in_scores=False):
    """Create a DuckDB with minimal tables for QC testing.

    Args:
        db_path:          Path for the temp DuckDB file.
        include_tables:   List of table names to create. If None, create all required.
        nulls_in_scores:  If True, insert NULLs in composite_score.
    """
    if include_tables is None:
        include_tables = list(REQUIRED_TABLES)

    con = duckdb.connect(str(db_path))

    if "buildings" in include_tables:
        con.execute("""
            CREATE TABLE buildings (
                building_id TEXT, centroid_lat DOUBLE,
                centroid_lon DOUBLE, area_m2 DOUBLE
            )
        """)
        con.execute("""
            INSERT INTO buildings VALUES
            ('B001', 37.95, 23.90, 120.0),
            ('B002', 38.01, 23.95, 85.0)
        """)

    if "features_terrain" in include_tables:
        con.execute("""
            CREATE TABLE features_terrain (
                building_id TEXT, elevation_m DOUBLE, slope_deg DOUBLE
            )
        """)
        con.execute("""
            INSERT INTO features_terrain VALUES
            ('B001', 150.0, 12.5), ('B002', 320.0, 25.0)
        """)

    if "features_vegetation" in include_tables:
        con.execute("CREATE TABLE features_vegetation (building_id TEXT)")
        con.execute("INSERT INTO features_vegetation VALUES ('B001'), ('B002')")

    if "features_fire_weather" in include_tables:
        con.execute("CREATE TABLE features_fire_weather (building_id TEXT)")
        con.execute("INSERT INTO features_fire_weather VALUES ('B001'), ('B002')")

    if "features_fire_history" in include_tables:
        con.execute("""
            CREATE TABLE features_fire_history (
                building_id TEXT, dist_to_nearest_fire_m DOUBLE
            )
        """)
        con.execute("""
            INSERT INTO features_fire_history VALUES
            ('B001', 2500.0), ('B002', 800.0)
        """)

    if "risk_scores" in include_tables:
        composite = "NULL" if nulls_in_scores else "0.65"
        con.execute("""
            CREATE TABLE risk_scores (
                building_id TEXT, composite_score DOUBLE,
                risk_class INTEGER, score_terrain DOUBLE,
                score_vegetation DOUBLE, score_fire_weather DOUBLE,
                score_fire_history DOUBLE
            )
        """)
        con.execute(f"""
            INSERT INTO risk_scores VALUES
            ('B001', {composite}, 4, 0.6, 0.7, 0.5, 0.8),
            ('B002', 0.42, 2, 0.3, 0.4, 0.6, 0.5)
        """)

    con.close()


# ---------------------------------------------------------------------------
# check_tables
# ---------------------------------------------------------------------------

class TestCheckTables:
    """Tests for the table existence check."""

    def test_all_tables_present(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db)
        con = duckdb.connect(str(db), read_only=True)
        missing = check_tables(con)
        con.close()
        assert missing == []

    def test_missing_one_table(self, tmp_path):
        db = tmp_path / "test.duckdb"
        tables = [t for t in REQUIRED_TABLES if t != "risk_scores"]
        _create_minimal_db(db, include_tables=tables)
        con = duckdb.connect(str(db), read_only=True)
        missing = check_tables(con)
        con.close()
        assert "risk_scores" in missing
        assert len(missing) == 1

    def test_empty_database(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, include_tables=[])
        con = duckdb.connect(str(db), read_only=True)
        missing = check_tables(con)
        con.close()
        assert set(missing) == set(REQUIRED_TABLES)

    def test_returns_list(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db)
        con = duckdb.connect(str(db), read_only=True)
        result = check_tables(con)
        con.close()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# check_columns
# ---------------------------------------------------------------------------

class TestCheckColumns:
    """Tests for the column presence check."""

    def test_all_columns_present(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db)
        con = duckdb.connect(str(db), read_only=True)
        missing = check_columns(con)
        con.close()
        assert missing == []

    def test_missing_column_detected(self, tmp_path):
        """Create buildings table without area_m2 — should detect it."""
        db = tmp_path / "test.duckdb"
        con = duckdb.connect(str(db))
        con.execute("""
            CREATE TABLE buildings (
                building_id TEXT, centroid_lat DOUBLE, centroid_lon DOUBLE
            )
        """)
        # Create other tables normally
        for t in REQUIRED_TABLES:
            if t != "buildings":
                con.execute(f"CREATE TABLE {t} (building_id TEXT)")
        # Add required columns to risk_scores
        con.execute("DROP TABLE risk_scores")
        con.execute("""
            CREATE TABLE risk_scores (
                building_id TEXT, composite_score DOUBLE,
                risk_class INTEGER, score_terrain DOUBLE,
                score_vegetation DOUBLE, score_fire_weather DOUBLE,
                score_fire_history DOUBLE
            )
        """)
        # Add required columns to features_terrain
        con.execute("DROP TABLE features_terrain")
        con.execute("""
            CREATE TABLE features_terrain (
                building_id TEXT, elevation_m DOUBLE, slope_deg DOUBLE
            )
        """)
        # Add required columns to features_fire_history
        con.execute("DROP TABLE features_fire_history")
        con.execute("""
            CREATE TABLE features_fire_history (
                building_id TEXT, dist_to_nearest_fire_m DOUBLE
            )
        """)
        con.close()

        con = duckdb.connect(str(db), read_only=True)
        missing = check_columns(con)
        con.close()
        assert "buildings.area_m2" in missing

    def test_missing_table_gracefully_skipped(self, tmp_path):
        """If a table is missing entirely, check_columns should skip it (not crash)."""
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, include_tables=["buildings"])
        con = duckdb.connect(str(db), read_only=True)
        # Should not raise, even though risk_scores/features_terrain etc. are missing
        missing = check_columns(con)
        con.close()
        assert isinstance(missing, list)


# ---------------------------------------------------------------------------
# check_nulls
# ---------------------------------------------------------------------------

class TestCheckNulls:
    """Tests for the NULL violation check on risk_scores."""

    def test_no_nulls(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, nulls_in_scores=False)
        con = duckdb.connect(str(db), read_only=True)
        violations = check_nulls(con)
        con.close()
        assert violations == []

    def test_null_detected(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, nulls_in_scores=True)
        con = duckdb.connect(str(db), read_only=True)
        violations = check_nulls(con)
        con.close()
        assert len(violations) > 0
        assert any("composite_score" in v for v in violations)

    def test_missing_table_reports_failure(self, tmp_path):
        """If risk_scores doesn't exist, check_nulls should report query failure."""
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, include_tables=["buildings"])
        con = duckdb.connect(str(db), read_only=True)
        violations = check_nulls(con)
        con.close()
        assert len(violations) == len(NON_NULL_SCORE_COLS)
        assert all("query failed" in v for v in violations)


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------

class TestRunAll:
    """Tests for the integrated run_all function."""

    def test_all_pass(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db)
        result = run_all(db, exit_on_failure=False)
        assert result is True

    def test_missing_db_file(self, tmp_path):
        db = tmp_path / "nonexistent.duckdb"
        result = run_all(db, exit_on_failure=False)
        assert result is False

    def test_missing_table_fails(self, tmp_path):
        db = tmp_path / "test.duckdb"
        tables = [t for t in REQUIRED_TABLES if t != "buildings"]
        _create_minimal_db(db, include_tables=tables)
        result = run_all(db, exit_on_failure=False)
        assert result is False

    def test_null_violation_fails(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _create_minimal_db(db, nulls_in_scores=True)
        result = run_all(db, exit_on_failure=False)
        assert result is False

    def test_exit_on_failure_default(self, tmp_path):
        """With exit_on_failure=True (default), missing DB should sys.exit."""
        db = tmp_path / "nonexistent.duckdb"
        with pytest.raises(SystemExit):
            run_all(db, exit_on_failure=True)
