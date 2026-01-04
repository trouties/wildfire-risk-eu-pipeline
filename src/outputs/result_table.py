"""
result_table.py — WildfireRisk-EU Output: Risk Score Table (Task T12)

Queries risk_scores JOIN buildings from DuckDB and exports a flat CSV
containing one row per building with all scores and coordinates.

Output: outputs/tables/risk_scores_attica.csv (84,767 rows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.qc.schema import run_all as qc_run_all  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "wildfire_risk.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_FILE = OUT_DIR / "risk_scores_attica.csv"
OUT_FILE_PARQUET = OUT_DIR / "risk_scores_attica.parquet"

COLUMNS_OUT = [
    "building_id",
    "centroid_lat",
    "centroid_lon",
    "area_m2",
    "composite_score",
    "risk_class",
    "score_terrain",
    "score_vegetation",
    "score_fire_weather",
    "score_fire_history",
]

QUERY = """
SELECT
    r.building_id,
    b.centroid_lat,
    b.centroid_lon,
    b.area_m2,
    r.composite_score,
    r.risk_class,
    r.score_terrain,
    r.score_vegetation,
    r.score_fire_weather,
    r.score_fire_history
FROM risk_scores r
JOIN buildings b USING (building_id)
ORDER BY r.building_id
"""


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run schema/integrity checks before exporting
    qc_run_all(DB_PATH)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(QUERY).df()
    con.close()

    # QC
    n_rows = len(df)
    null_score = df["composite_score"].isna().sum()
    null_class = df["risk_class"].isna().sum()

    if n_rows == 0:
        sys.exit("ERROR result_table: no rows returned from database")
    if null_score > 0 or null_class > 0:
        sys.exit(
            f"ERROR result_table: {null_score} null composite_score, "
            f"{null_class} null risk_class"
        )

    df[COLUMNS_OUT].to_csv(OUT_FILE, index=False)
    df[COLUMNS_OUT].to_parquet(OUT_FILE_PARQUET, index=False)

    print(f"[result_table] Wrote {n_rows:,} rows -> {OUT_FILE}")
    print(f"[result_table] Wrote {n_rows:,} rows -> {OUT_FILE_PARQUET}")
    print(
        "[result_table] risk_class distribution:\n"
        + df["risk_class"].value_counts().sort_index().to_string()
    )
    return OUT_FILE


if __name__ == "__main__":
    main()
