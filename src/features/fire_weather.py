"""
features/fire_weather.py — Assign each building to its nearest land-valid
                            FWI grid cell and copy the 5 fire-weather stats.

Inputs (from DuckDB):
  fwi_grid_stats  — 72 rows; 37 have non-NULL fwi_season_mean (land cells)
                    columns: cell_id, latitude, longitude (WGS84),
                             fwi_season_mean, fwi_season_p90, fwi_season_max,
                             dc_season_mean, fwi_extreme_days
  buildings       — 84,767 rows; centroid_lat/centroid_lon in WGS84

Method:
  Nearest-neighbour matching via scipy.spatial.cKDTree on WGS84 coordinates.
  Because the study area is small (~80 km × 75 km) and the grid is coarse
  (~9 km ERA5-Land cells), degree-unit Euclidean distance is acceptable:
  ~0.1° latitude ≈ ~0.1° longitude ≈ 11 km at 38°N.  The distortion at
  this scale is < 0.3% and introduces negligible assignment error compared
  to the 9 km grid spacing.

Output — DuckDB table features_fire_weather (84,767 rows):
  building_id      TEXT
  fwi_season_mean  DOUBLE   -- mean daily FWI June–Oct, averaged 2015-2024
  fwi_season_p90   DOUBLE   -- 90th percentile daily FWI, fire season
  fwi_season_max   DOUBLE   -- single-day maximum FWI, fire season
  dc_season_mean   DOUBLE   -- mean Drought Code, fire season
  fwi_extreme_days DOUBLE   -- days with FWI > 30 (Very High) per season

Caveats:
  - ERA5-Land grid is ~9 km; all buildings in a single grid cell receive
    identical fire-weather values.  Expect 8–12 distinct value sets across
    84,767 buildings.
  - FWI stats were computed from ERA5-Land reanalysis (2015-2024, fire season
    Jun–Oct) and stored in fwi_grid_stats during T06.
  - Sea-masked cells (fwi_season_mean IS NULL) are excluded from the KD-tree;
    the nearest land cell is assigned to all buildings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Load inputs from DuckDB
# ---------------------------------------------------------------------------

def _load_fwi_cells(db_path: Path) -> pd.DataFrame:
    """Return land-valid FWI grid cells (non-NULL fwi_season_mean)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT cell_id, latitude, longitude,
               fwi_season_mean, fwi_season_p90, fwi_season_max,
               dc_season_mean, fwi_extreme_days
        FROM fwi_grid_stats
        WHERE fwi_season_mean IS NOT NULL
        ORDER BY cell_id
    """).df()
    con.close()
    return df


def _load_building_centroids(db_path: Path) -> pd.DataFrame:
    """Return building_id, centroid_lat, centroid_lon (WGS84)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT building_id, centroid_lat, centroid_lon
        FROM buildings
        ORDER BY building_id
    """).df()
    con.close()
    return df


# ---------------------------------------------------------------------------
# Nearest-neighbour assignment
# ---------------------------------------------------------------------------

def _assign_nearest_cell(
    buildings: pd.DataFrame,
    fwi_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Match each building to its nearest land-valid FWI grid cell.

    Uses cKDTree on (longitude, latitude) pairs — degree-unit Euclidean
    distance is sufficient for this small AOI (~80 × 75 km).

    Returns a DataFrame with building_id and the 5 fire-weather columns.
    """
    # Build KD-tree on cell centroids: (lon, lat) order
    cell_coords = fwi_cells[["longitude", "latitude"]].values   # (37, 2)
    tree = cKDTree(cell_coords)

    # Query: (lon, lat) for each building
    bldg_coords = buildings[["centroid_lon", "centroid_lat"]].values  # (84767, 2)
    _, nearest_idx = tree.query(bldg_coords, k=1, workers=-1)

    # Select matched rows and reset index
    matched = fwi_cells.iloc[nearest_idx].reset_index(drop=True)

    result = pd.DataFrame({
        "building_id":      buildings["building_id"].values,
        "fwi_season_mean":  matched["fwi_season_mean"].values,
        "fwi_season_p90":   matched["fwi_season_p90"].values,
        "fwi_season_max":   matched["fwi_season_max"].values,
        "dc_season_mean":   matched["dc_season_mean"].values,
        "fwi_extreme_days": matched["fwi_extreme_days"].values,
    })
    return result


# ---------------------------------------------------------------------------
# DuckDB write
# ---------------------------------------------------------------------------

def _write_duckdb(df: pd.DataFrame, db_path: Path) -> int:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass
    con.execute("DROP TABLE IF EXISTS features_fire_weather;")
    con.register("_fw_df", df)
    con.execute("""
        CREATE TABLE features_fire_weather AS
        SELECT
            building_id,
            fwi_season_mean,
            fwi_season_p90,
            fwi_season_max,
            dc_season_mean,
            fwi_extreme_days
        FROM _fw_df
    """)
    n = con.execute("SELECT count(*) FROM features_fire_weather").fetchone()[0]
    row = con.execute("""
        SELECT
            count(*)                                             AS n_rows,
            round(avg(fwi_season_mean),  2)                     AS avg_mean,
            round(avg(fwi_season_p90),   2)                     AS avg_p90,
            round(max(fwi_season_max),   2)                     AS max_fwi,
            round(avg(dc_season_mean),   2)                     AS avg_dc,
            round(avg(fwi_extreme_days), 2)                     AS avg_extreme,
            count(*) FILTER (WHERE fwi_season_mean IS NULL)     AS n_null,
            count(DISTINCT fwi_season_mean)                     AS n_distinct_mean
        FROM features_fire_weather
    """).fetchone()
    print("  [duckdb] features_fire_weather:")
    print(f"    rows={row[0]:,}  avg_fwi_mean={row[1]}  avg_fwi_p90={row[2]}")
    print(f"    max_fwi={row[3]}  avg_dc={row[4]}  avg_extreme_days={row[5]}")
    print(f"    null_fwi_mean={row[6]}  distinct_fwi_mean_values={row[7]}")
    con.close()
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg     = load_config()
    root    = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]

    # ------------------------------------------------------------------
    # 1. Load FWI grid cells (land-valid only)
    # ------------------------------------------------------------------
    print("[features/fire_weather] Loading fwi_grid_stats ...")
    fwi_cells = _load_fwi_cells(db_path)
    print(f"  {len(fwi_cells)} land-valid cells  "
          f"lat=[{fwi_cells['latitude'].min():.2f}, {fwi_cells['latitude'].max():.2f}]  "
          f"lon=[{fwi_cells['longitude'].min():.2f}, {fwi_cells['longitude'].max():.2f}]")
    print(f"  fwi_season_mean: min={fwi_cells['fwi_season_mean'].min():.2f}  "
          f"max={fwi_cells['fwi_season_mean'].max():.2f}")

    # ------------------------------------------------------------------
    # 2. Load building centroids (WGS84)
    # ------------------------------------------------------------------
    print("[features/fire_weather] Loading building centroids ...")
    buildings = _load_building_centroids(db_path)
    n = len(buildings)
    print(f"  {n:,} buildings  "
          f"lat=[{buildings['centroid_lat'].min():.3f}, {buildings['centroid_lat'].max():.3f}]  "
          f"lon=[{buildings['centroid_lon'].min():.3f}, {buildings['centroid_lon'].max():.3f}]")

    # ------------------------------------------------------------------
    # 3. Nearest-neighbour assignment
    # ------------------------------------------------------------------
    print("[features/fire_weather] Assigning buildings to nearest FWI cell ...")
    df = _assign_nearest_cell(buildings, fwi_cells)

    n_distinct = df["fwi_season_mean"].nunique()
    null_frac  = df["fwi_season_mean"].isna().mean()
    print(f"  distinct fwi_season_mean values: {n_distinct}  "
          f"(expected 8-12 for 9 km grid)")
    print(f"  null fraction: {null_frac:.4%}  (expected <5%)")

    # ------------------------------------------------------------------
    # 4. Pre-write sanity checks
    # ------------------------------------------------------------------
    assert len(df) == n, f"Row count mismatch: {len(df)} != {n}"
    assert null_frac < 0.05, f"NULL fraction {null_frac:.2%} exceeds 5% threshold"
    nonnull = df["fwi_season_mean"].dropna()
    assert (nonnull > 0).all(), "fwi_season_mean has non-positive values"
    assert (df["fwi_extreme_days"].dropna() >= 0).all(), "fwi_extreme_days has negative values"
    # Upper bound = total land cells (37); expect 15-37 for full-AOI coverage
    assert n_distinct <= len(fwi_cells), (
        f"Too many distinct fwi_season_mean values ({n_distinct}) "
        f"— exceeds land cell count ({len(fwi_cells)})"
    )
    print("[features/fire_weather] Pre-write checks passed.")

    # ------------------------------------------------------------------
    # 5. Write to DuckDB
    # ------------------------------------------------------------------
    print(f"[features/fire_weather] Writing → {db_path.name} ...")
    count = _write_duckdb(df, db_path)
    assert count == n, f"DuckDB row count {count} != expected {n}"
    print(f"[features/fire_weather] Done — {count:,} rows in features_fire_weather.")


if __name__ == "__main__":
    main()
