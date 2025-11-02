"""
db.py — DuckDB connection management and schema initialisation.

The pipeline uses a single DuckDB file for all tabular data.
This module owns the CREATE TABLE DDL and provides a connection helper.

Tables
------
buildings            Core exposure table: one row per building.
terrain_features     6 terrain-derived features per building.
vegetation_features  9 vegetation features per building.
fire_weather_features 5 fire weather features per building.
fire_history_features 6 fire history features per building.
proximity_features   4 proximity features per building.
risk_scores          Composite score + component scores + risk class.
fire_events          EFFIS fire perimeters (reference geometry table).
validation_results   Per-building validation labels and scores.
pipeline_metadata    Run provenance: timestamps, data versions, checksums.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

# ---------------------------------------------------------------------------
# DDL — one CREATE TABLE IF NOT EXISTS per table
# ---------------------------------------------------------------------------

_DDL = """
-- Requires DuckDB spatial extension for GEOMETRY type
INSTALL spatial;
LOAD spatial;

-- ── Core exposure ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS buildings (
    building_id     VARCHAR PRIMARY KEY,   -- OSM way/relation ID or MSFT ID
    source          VARCHAR NOT NULL,      -- 'osm' | 'msft'
    osm_id          BIGINT,
    building_type   VARCHAR,               -- OSM building= tag value
    building_use    VARCHAR,               -- residential / commercial / critical
    footprint_m2    DOUBLE,
    centroid_lon    DOUBLE NOT NULL,       -- WGS84
    centroid_lat    DOUBLE NOT NULL,
    centroid_x      DOUBLE,               -- EPSG:2100 easting (m)
    centroid_y      DOUBLE,               -- EPSG:2100 northing (m)
    geometry        GEOMETRY,             -- footprint polygon, EPSG:2100
    data_version    VARCHAR,
    loaded_at       TIMESTAMP DEFAULT current_timestamp
);

-- ── Terrain features ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS terrain_features (
    building_id       VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    elevation_m       DOUBLE,
    slope_deg         DOUBLE,
    aspect_deg        DOUBLE,
    south_aspect_score DOUBLE,            -- derived: abs(cos(aspect_rad))
    tpi_300m          DOUBLE,
    tri_300m          DOUBLE,
    computed_at       TIMESTAMP DEFAULT current_timestamp
);

-- ── Vegetation features ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vegetation_features (
    building_id          VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    ndvi_mean_100m       DOUBLE,
    ndvi_mean_500m       DOUBLE,
    ndvi_max_500m        DOUBLE,
    veg_fraction_100m    DOUBLE,          -- [0, 1]
    veg_fraction_500m    DOUBLE,          -- [0, 1]
    dist_to_forest_m     DOUBLE,
    dist_to_scrubland_m  DOUBLE,
    wui_class            INTEGER,         -- 0=urban, 1=interface, 2=intermix
    veg_continuity_500m  DOUBLE,          -- [0, 1]
    computed_at          TIMESTAMP DEFAULT current_timestamp
);

-- ── Fire weather features ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fire_weather_features (
    building_id        VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    scoring_period     VARCHAR NOT NULL,  -- e.g. '2015-2024' or '2015-2017' for pre-Mati
    fwi_season_mean    DOUBLE,
    fwi_season_p90     DOUBLE,
    fwi_season_max     DOUBLE,
    dc_season_mean     DOUBLE,
    fwi_extreme_days   DOUBLE,            -- count of days FWI > 30
    computed_at        TIMESTAMP DEFAULT current_timestamp
);

-- ── Fire history features ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fire_history_features (
    building_id              VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    history_cutoff           DATE,        -- features computed using data up to this date
    dist_to_nearest_fire_m   DOUBLE,
    fire_count_5km           INTEGER,
    fire_count_10km          INTEGER,
    ever_burned              BOOLEAN,
    firms_hotspot_count_5km  INTEGER,
    recency_score            DOUBLE,
    computed_at              TIMESTAMP DEFAULT current_timestamp
);

-- ── Proximity features ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS proximity_features (
    building_id              VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    dist_to_road_m           DOUBLE,
    dist_to_fire_station_m   DOUBLE,
    road_density_500m        DOUBLE,      -- m / km²
    building_density_500m    DOUBLE,      -- count
    computed_at              TIMESTAMP DEFAULT current_timestamp
);

-- ── Risk scores ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_scores (
    building_id           VARCHAR PRIMARY KEY REFERENCES buildings(building_id),
    scoring_run_id        VARCHAR NOT NULL,   -- links to pipeline_metadata
    terrain_score         DOUBLE,             -- normalised group score [0, 1]
    vegetation_score      DOUBLE,
    fire_weather_score    DOUBLE,
    fire_history_score    DOUBLE,
    proximity_score       DOUBLE,
    composite_score       DOUBLE NOT NULL,    -- weighted sum [0, 1]
    risk_class            INTEGER NOT NULL,   -- 1 (Very Low) to 5 (Very High)
    risk_label            VARCHAR NOT NULL,
    scored_at             TIMESTAMP DEFAULT current_timestamp
);

-- ── Fire events (EFFIS perimeters) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fire_events (
    event_id       VARCHAR PRIMARY KEY,
    source         VARCHAR NOT NULL,       -- 'effis' | 'emsr300' | 'mcd64a1'
    fire_name      VARCHAR,
    country        VARCHAR,
    date_start     DATE,
    date_end       DATE,
    area_ha        DOUBLE,
    geometry       GEOMETRY,              -- fire perimeter polygon, EPSG:2100
    notes          VARCHAR,
    loaded_at      TIMESTAMP DEFAULT current_timestamp
);

-- ── Validation results ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS validation_results (
    building_id           VARCHAR REFERENCES buildings(building_id),
    event_id              VARCHAR REFERENCES fire_events(event_id),
    burned                BOOLEAN NOT NULL,          -- ground truth
    composite_score       DOUBLE,                    -- score using pre-event features
    risk_class            INTEGER,
    baseline_score        DOUBLE,                    -- dist_to_forest baseline
    is_false_negative     BOOLEAN,                   -- burned=True but risk_class <= 2
    is_false_positive     BOOLEAN,                   -- burned=False but risk_class >= 4
    PRIMARY KEY (building_id, event_id)
);

-- ── Pipeline metadata ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_metadata (
    run_id          VARCHAR PRIMARY KEY,
    pipeline_version VARCHAR,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    aoi_bbox        VARCHAR,              -- JSON string [west,south,east,north]
    config_hash     VARCHAR,              -- MD5 of merged config YAML
    building_count  INTEGER,
    scored_count    INTEGER,
    notes           VARCHAR
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection.  Caller is responsible for closing it."""
    return duckdb.connect(str(db_path), read_only=read_only)


@contextmanager
def db_connection(
    db_path: str | Path, read_only: bool = False
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager: open → yield → close DuckDB connection."""
    conn = get_connection(db_path, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(db_path: str | Path) -> None:
    """Create all tables (IF NOT EXISTS) in the target DuckDB file.

    Safe to call multiple times — existing tables and data are untouched.
    """
    with db_connection(db_path) as conn:
        conn.executescript(_DDL)
    print(f"Schema initialised at: {db_path}")


if __name__ == "__main__":
    from src.utils.config import load_config, resolve_path

    cfg = load_config()
    db_path = resolve_path(cfg["pipeline"]["paths"]["db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_schema(db_path)
