"""
features/terrain.py — Sample terrain rasters at building centroids and write
                      the features_terrain table to DuckDB.

Inputs (all EPSG:2100, 30 m float32):
  data/processed/dem_attica_epsg2100.tif
  data/processed/slope_attica_epsg2100.tif
  data/processed/aspect_attica_epsg2100.tif
  data/processed/tpi_attica_epsg2100.tif
  data/processed/tri_attica_epsg2100.tif
  data/processed/buildings_attica_epsg2100.gpkg

Output:
  data/wildfire_risk.duckdb → features_terrain
    building_id TEXT, elevation_m DOUBLE, slope_deg DOUBLE, aspect_deg DOUBLE,
    south_aspect_score DOUBLE, tpi_300m DOUBLE, tri_300m DOUBLE
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODATA_SENTINEL = -9999.0   # nodata value written by preprocess/terrain.py


# ---------------------------------------------------------------------------
# Raster sampling
# ---------------------------------------------------------------------------

def _sample_raster(raster_path: Path, coords: list[tuple[float, float]]) -> np.ndarray:
    """Sample a single-band raster at *coords* ((x, y) in raster CRS).

    Returns a 1-D float64 array of length len(coords).  Nodata pixels
    (== NODATA_SENTINEL) are replaced with np.nan.

    rasterio.DatasetReader.sample() uses the raster transform to convert
    world coordinates to pixel indices — no CRS transform needed because
    both building centroids and the rasters are in EPSG:2100.
    """
    with rasterio.open(raster_path) as src:
        # src.sample() yields one array per coordinate; each array has shape (n_bands,).
        # With a single-band raster, each element is shape (1,), so the stacked
        # result is (n_coords, 1) — squeeze to 1-D.
        raw = np.array(list(src.sample(coords)), dtype="float64").squeeze(-1)

    # Replace nodata sentinel with NaN
    raw[raw == NODATA_SENTINEL] = np.nan
    return raw


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def _south_aspect_score(aspect_deg: np.ndarray) -> np.ndarray:
    """Return |cos(aspect_rad)| ∈ [0, 1].

    Convention used in preprocess/terrain.py: 0° = North, 180° = South,
    clockwise.  cos(0°)=1 (north), cos(180°)=-1 (south), abs() maps both
    N and S to 1 — but fire risk is highest on *south*-facing slopes in
    the Northern Hemisphere (drier, more insolation).

    Correct south-face formula: score = (1 − cos(aspect_rad)) / 2
      → 0 at N (0°), 1 at S (180°), 0.5 at E/W (90°/270°)

    This matches the wildfire literature (south/southwest exposure = drier
    fuels) and is bounded [0, 1] with no ambiguity at aspect = 0 (north).

    NaN aspect → NaN score.
    """
    with np.errstate(invalid="ignore"):
        score = (1.0 - np.cos(np.radians(aspect_deg))) / 2.0
    return score


# ---------------------------------------------------------------------------
# DuckDB write
# ---------------------------------------------------------------------------

def _write_features_terrain(df: pd.DataFrame, db_path: Path) -> int:
    """Write *df* to the features_terrain table in the existing DuckDB database.

    Uses the same register-then-CREATE-AS pattern as T05/T06 to avoid any
    DuckDB storage-version issues with native GEOMETRY types.
    """
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass
    con.execute("INSTALL spatial; LOAD spatial;")

    con.execute("DROP TABLE IF EXISTS features_terrain;")
    con.register("_terrain_df", df)
    con.execute("""
        CREATE TABLE features_terrain AS
        SELECT
            building_id,
            elevation_m,
            slope_deg,
            aspect_deg,
            south_aspect_score,
            tpi_300m,
            tri_300m
        FROM _terrain_df
    """)

    count = con.execute("SELECT count(*) FROM features_terrain").fetchone()[0]

    # Sanity stats
    row = con.execute("""
        SELECT
            count(*)                                     AS n,
            round(avg(slope_deg), 3)                     AS avg_slope,
            round(avg(elevation_m), 1)                   AS avg_elev,
            round(min(tpi_300m), 2)                      AS min_tpi,
            round(max(tpi_300m), 2)                      AS max_tpi,
            round(avg(south_aspect_score), 4)            AS avg_sas,
            count(*) FILTER (WHERE elevation_m IS NULL)  AS null_elev
        FROM features_terrain
    """).fetchone()
    print("  [duckdb] features_terrain sanity:")
    print(f"    rows={row[0]:,}  avg_slope={row[1]}°  avg_elev={row[2]} m")
    print(f"    tpi range=[{row[3]}, {row[4]}]  avg_south_aspect_score={row[5]}")
    print(f"    elevation NULL count={row[6]:,} ({row[6]/row[0]*100:.2f}%)")

    con.close()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")
    proc = root / "data/processed"
    db_path = root / cfg["pipeline"]["paths"]["db"]

    gpkg_path = proc / "buildings_attica_epsg2100.gpkg"
    raster_specs: list[tuple[str, Path]] = [
        ("elevation_m",  proc / "dem_attica_epsg2100.tif"),
        ("slope_deg",    proc / "slope_attica_epsg2100.tif"),
        ("aspect_deg",   proc / "aspect_attica_epsg2100.tif"),
        ("tpi_300m",     proc / "tpi_attica_epsg2100.tif"),
        ("tri_300m",     proc / "tri_attica_epsg2100.tif"),
    ]

    # ------------------------------------------------------------------
    # 1. Load building centroids (EPSG:2100)
    # ------------------------------------------------------------------
    print(f"[features/terrain] Reading buildings from {gpkg_path.name} ...")
    gdf = gpd.read_file(gpkg_path)
    n_buildings = len(gdf)
    print(f"  buildings loaded: {n_buildings:,}")

    # Compute centroids in EPSG:2100 directly from polygon geometry.
    # Do NOT use the centroid_lat/lon columns — those are WGS84 and cannot
    # be used as coordinates for EPSG:2100 raster sampling.
    centroids = gdf.geometry.centroid
    coords = list(zip(centroids.x, centroids.y))

    # ------------------------------------------------------------------
    # 2. Sample all five rasters
    # ------------------------------------------------------------------
    sampled: dict[str, np.ndarray] = {}
    for col_name, raster_path in raster_specs:
        print(f"[features/terrain] Sampling {raster_path.name} → '{col_name}' ...")
        sampled[col_name] = _sample_raster(raster_path, coords)
        valid_count = np.sum(~np.isnan(sampled[col_name]))
        print(f"  valid pixels: {valid_count:,} / {n_buildings:,}")

    # ------------------------------------------------------------------
    # 3. Derive south_aspect_score
    # ------------------------------------------------------------------
    print("[features/terrain] Deriving south_aspect_score ...")
    sampled["south_aspect_score"] = _south_aspect_score(sampled["aspect_deg"])

    # ------------------------------------------------------------------
    # 4. Build DataFrame with explicit column order
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "building_id":       gdf["building_id"].values,
        "elevation_m":       sampled["elevation_m"],
        "slope_deg":         sampled["slope_deg"],
        "aspect_deg":        sampled["aspect_deg"],
        "south_aspect_score": sampled["south_aspect_score"],
        "tpi_300m":          sampled["tpi_300m"],
        "tri_300m":          sampled["tri_300m"],
    })

    # ------------------------------------------------------------------
    # 5. Acceptance-criteria checks before writing
    # ------------------------------------------------------------------
    slope_valid = df["slope_deg"].dropna()
    sas_valid   = df["south_aspect_score"].dropna()
    null_frac   = df["elevation_m"].isna().mean()

    assert len(df) == n_buildings, f"Row count mismatch: {len(df)} != {n_buildings}"
    assert slope_valid.between(0, 90).all(), \
        f"slope_deg out of [0, 90]: min={slope_valid.min():.2f} max={slope_valid.max():.2f}"
    assert sas_valid.between(0, 1).all(), \
        f"south_aspect_score out of [0, 1]: min={sas_valid.min():.4f} max={sas_valid.max():.4f}"
    assert null_frac < 0.05, \
        f"elevation_m NULL fraction {null_frac:.3f} exceeds 5% threshold"
    assert df["tpi_300m"].dropna().min() < 0 and df["tpi_300m"].dropna().max() > 0, \
        "tpi_300m has no ridge/valley contrast (both min and max not opposite sign)"

    print("[features/terrain] Pre-write checks passed.")

    # ------------------------------------------------------------------
    # 6. Write to DuckDB
    # ------------------------------------------------------------------
    print(f"[features/terrain] Writing features_terrain → {db_path.name} ...")
    count = _write_features_terrain(df, db_path)

    assert count == n_buildings, f"DuckDB row count {count} != expected {n_buildings}"
    print(f"[features/terrain] Done — {count:,} rows in features_terrain.")


if __name__ == "__main__":
    main()
