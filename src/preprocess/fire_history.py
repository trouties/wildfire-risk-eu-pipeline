"""
preprocess/fire_history.py — Reproject and load EFFIS perimeters and FIRMS hotspots.

EFFIS perimeters are polygon data; FIRMS detections are point data.  Both are
reprojected to EPSG:2100 and saved as GeoPackages so T09 can do spatial queries
(distance, intersection, counting) against them.

Inputs:
  data/raw/effis/fire_perimeters_greece.gpkg     (27 polygons, EPSG:4326)
  data/raw/firms/viirs_attica_2015_2024.csv      (6130 VIIRS hotspots)

Outputs:
  data/processed/effis_perimeters_attica_epsg2100.gpkg   (14 perimeters in AOI + buffer)
  data/processed/firms_viirs_attica_epsg2100.gpkg        (all hotspots, EPSG:2100 points)
  data/wildfire_risk.duckdb → effis_perimeters           WKT polygon table
  data/wildfire_risk.duckdb → firms_hotspots             WKT point table
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gpkg_to_duckdb(gdf: gpd.GeoDataFrame, table_name: str,
                          col_order: list[str], con: duckdb.DuckDBPyConnection) -> int:
    """Insert a GeoDataFrame into DuckDB using WKT geometry."""
    df_db = gdf[col_order].copy()
    df_db["geometry"] = gdf.geometry.to_wkt()
    # Replace the geometry column (already set above) in col_order position
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.register(f"_{table_name}_df", df_db)
    cols_sql = ", ".join(col_order)
    con.execute(f"CREATE TABLE {table_name} AS SELECT {cols_sql} FROM _{table_name}_df")
    return con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]


# ---------------------------------------------------------------------------
# EFFIS
# ---------------------------------------------------------------------------

def process_effis(root: Path, cfg: dict) -> gpd.GeoDataFrame:
    """Clip EFFIS perimeters to AOI + buffer, reproject to EPSG:2100."""
    raw_path = root / "data/raw/effis/fire_perimeters_greece.gpkg"
    out_path = root / "data/processed/effis_perimeters_attica_epsg2100.gpkg"
    crs_work = cfg["pipeline"]["aoi"]["crs_working"]
    bbox     = cfg["pipeline"]["aoi"]["bbox"]  # [W, S, E, N] in WGS84
    buffer_deg = 0.2   # ~20 km buffer around AOI to capture nearby perimeters

    print(f"[effis] Reading {raw_path.name} ...")
    gdf = gpd.read_file(raw_path)
    print(f"  total rows: {len(gdf)}  CRS: {gdf.crs}")

    # Clip to expanded AOI (AOI + buffer so edge fires are retained)
    W, S, E, N = bbox
    gdf_aoi = gdf.cx[W - buffer_deg : E + buffer_deg, S - buffer_deg : N + buffer_deg].copy()
    print(f"  after AOI+buffer clip: {len(gdf_aoi)} perimeters")

    # Reproject
    gdf_aoi = gdf_aoi.to_crs(crs_work)

    # Ensure event_date is string for portability
    gdf_aoi["event_date"] = gdf_aoi["event_date"].astype(str)

    # Add geometry area in m² (post-reprojection)
    gdf_aoi["perimeter_area_m2"] = gdf_aoi.geometry.area

    # Sort by date
    gdf_aoi = gdf_aoi.sort_values("event_date").reset_index(drop=True)

    print(f"[effis] Writing → {out_path.name} ...")
    gdf_aoi.to_file(out_path, driver="GPKG", layer="fire_perimeters")

    # QC
    print(f"  years represented: {sorted(gdf_aoi['year'].unique())}")
    print(f"  area range: {gdf_aoi['area_ha'].min():.0f} – {gdf_aoi['area_ha'].max():.0f} ha")

    return gdf_aoi


# ---------------------------------------------------------------------------
# FIRMS
# ---------------------------------------------------------------------------

def process_firms(root: Path, cfg: dict) -> gpd.GeoDataFrame:
    """Read FIRMS CSV, build GeoDataFrame, reproject to EPSG:2100."""
    raw_path = root / "data/raw/firms/viirs_attica_2015_2024.csv"
    out_path = root / "data/processed/firms_viirs_attica_epsg2100.gpkg"
    crs_work = cfg["pipeline"]["aoi"]["crs_working"]

    print(f"[firms] Reading {raw_path.name} ...")
    df = pd.read_csv(raw_path)
    print(f"  raw rows: {len(df)}  confidence: {df['confidence'].unique().tolist()}")

    # Parse date
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")

    # Build GeoDataFrame from lat/lon
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Reproject to EPSG:2100
    gdf = gdf.to_crs(crs_work)

    # Assign sequential hotspot_id
    gdf = gdf.reset_index(drop=True)
    gdf.insert(0, "hotspot_id", [f"H{i:06d}" for i in range(len(gdf))])

    # Keep only relevant columns (drop redundant lat/lon in favour of geometry)
    keep_cols = [
        "hotspot_id", "geometry",
        "acq_date", "confidence", "frp",
        "bright_ti4", "daynight",
        "latitude", "longitude",   # original WGS84 coords retained for reference
    ]
    gdf = gdf[keep_cols].copy()

    print(f"[firms] Writing → {out_path.name} ...")
    gdf.to_file(out_path, driver="GPKG", layer="hotspots")

    # QC
    print(f"  date range: {gdf['acq_date'].min()} – {gdf['acq_date'].max()}")
    conf_counts = gdf["confidence"].value_counts().to_dict()
    print(f"  confidence counts: {conf_counts}")
    print(f"  FRP range: {gdf['frp'].min():.2f} – {gdf['frp'].max():.2f} MW")

    return gdf


# ---------------------------------------------------------------------------
# DuckDB loading
# ---------------------------------------------------------------------------

def load_to_duckdb(gdf_effis: gpd.GeoDataFrame, gdf_firms: gpd.GeoDataFrame,
                   db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass
    con.execute("INSTALL spatial; LOAD spatial;")

    # EFFIS
    effis_cols = ["fire_id", "geometry", "event_date", "year",
                  "region", "area_ha", "area_km2", "perimeter_area_m2"]
    n_effis = _load_gpkg_to_duckdb(gdf_effis, "effis_perimeters", effis_cols, con)
    print(f"  [duckdb] effis_perimeters: {n_effis} rows")

    # FIRMS
    firms_cols = ["hotspot_id", "geometry", "acq_date", "confidence",
                  "frp", "bright_ti4", "daynight", "latitude", "longitude"]
    n_firms = _load_gpkg_to_duckdb(gdf_firms, "firms_hotspots", firms_cols, con)
    print(f"  [duckdb] firms_hotspots : {n_firms} rows")

    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]

    gdf_effis = process_effis(root, cfg)
    print()
    gdf_firms = process_firms(root, cfg)
    print()

    print(f"[fire_history] Loading into DuckDB → {db_path.name} ...")
    load_to_duckdb(gdf_effis, gdf_firms, db_path)

    print("\n[fire_history] Done.")


if __name__ == "__main__":
    main()
