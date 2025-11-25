"""
preprocess/buildings.py — Clean, reproject, and load buildings into DuckDB.

Inputs:
  data/raw/buildings_osm_attica.gpkg
Outputs:
  data/processed/buildings_attica_epsg2100.gpkg   (GeoPackage, EPSG:2100)
  data/wildfire_risk.duckdb                        (buildings table)
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_by_centroid(gdf: gpd.GeoDataFrame, tolerance_m: float = 5.0) -> gpd.GeoDataFrame:
    """Drop duplicate buildings whose centroids are within *tolerance_m* metres.

    Strategy: snap centroids to a regular grid at *tolerance_m* spacing and
    keep the building with the largest area in each grid cell.  O(n log n).
    """
    centroids = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["_cx"] = centroids.x
    gdf["_cy"] = centroids.y
    gdf["_gx"] = (gdf["_cx"] / tolerance_m).round().astype(int)
    gdf["_gy"] = (gdf["_cy"] / tolerance_m).round().astype(int)

    before = len(gdf)
    gdf = (
        gdf.sort_values("area_m2", ascending=False)
        .drop_duplicates(subset=["_gx", "_gy"])
        .drop(columns=["_cx", "_cy", "_gx", "_gy"])
    )
    removed = before - len(gdf)
    if removed:
        print(f"  [dedup] removed {removed} near-duplicate buildings (tolerance={tolerance_m}m)")
    return gdf


def _load_duckdb(gdf: gpd.GeoDataFrame, gpkg_path: Path, db_path: Path) -> None:
    """Initialize DuckDB and populate the buildings table from the GeoPackage.

    Geometry is stored as WKT text so the table is portable across DuckDB storage
    versions.  Downstream spatial queries use ST_GeomFromText(geometry).
    """
    import pandas as pd

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale db file so we always start with a fresh, correctly-versioned db
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))

    # Enable latest storage format before writing any data, to support GEOMETRY
    # types with CRS identifiers (requires DuckDB spatial ≥ v1.5.0 storage).
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass  # older DuckDB versions don't have this pragma — ignore

    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("DROP TABLE IF EXISTS buildings;")

    # Build a plain pandas DataFrame — geometry stored as WKT text.
    # Using WKT avoids any DuckDB storage-version issues with native GEOMETRY.
    # Downstream can reconstruct geometry with ST_GeomFromText(geometry).
    df_db = pd.DataFrame({
        "building_id": gdf["building_id"].values,
        "geometry":    gdf.geometry.to_wkt(),   # WKT string, EPSG:2100 polygon
        "centroid_lat": gdf["centroid_lat"].values,
        "centroid_lon": gdf["centroid_lon"].values,
        "area_m2":     gdf["area_m2"].values.astype(float),
    })

    con.register("_buildings_df", df_db)
    con.execute("""
        CREATE TABLE buildings AS
        SELECT
            building_id,
            geometry,          -- WKT; use ST_GeomFromText(geometry) for spatial ops
            centroid_lat,
            centroid_lon,
            area_m2
        FROM _buildings_df
    """)

    count    = con.execute("SELECT count(*)    FROM buildings").fetchone()[0]
    avg_area = con.execute("SELECT avg(area_m2) FROM buildings").fetchone()[0]
    print(f"  [duckdb] buildings table: {count:,} rows | avg area = {avg_area:.1f} m²")

    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")

    in_path  = root / "data/raw/buildings_osm_attica.gpkg"
    out_path = root / "data/processed/buildings_attica_epsg2100.gpkg"
    db_path  = root / cfg["pipeline"]["paths"]["db"]
    crs_work = cfg["pipeline"]["aoi"]["crs_working"]    # EPSG:2100
    min_area = 5.0  # m² — remove trivial slivers

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read
    # ------------------------------------------------------------------
    print(f"[buildings] Reading {in_path.name} ...")
    gdf = gpd.read_file(in_path)
    print(f"  raw count : {len(gdf):,}")

    # ------------------------------------------------------------------
    # 2. Reproject to working CRS
    # ------------------------------------------------------------------
    print(f"[buildings] Reprojecting to {crs_work} ...")
    gdf = gdf.to_crs(crs_work)

    # ------------------------------------------------------------------
    # 3. Repair invalid geometries
    # ------------------------------------------------------------------
    invalid = ~gdf.geometry.is_valid
    n_invalid = invalid.sum()
    if n_invalid:
        print(f"  repairing {n_invalid} invalid geometries ...")
        import shapely
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(
            shapely.make_valid
        )

    # Drop empty / null geometries
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # ------------------------------------------------------------------
    # 4. Compute footprint area (m², valid after reprojection to meters)
    # ------------------------------------------------------------------
    gdf["area_m2"] = gdf.geometry.area.astype("float32")

    # Remove slivers
    before = len(gdf)
    gdf = gdf[gdf["area_m2"] >= min_area].copy()
    print(f"  after area filter (≥{min_area} m²): {len(gdf):,}  (dropped {before - len(gdf)})")

    # ------------------------------------------------------------------
    # 5. Deduplicate by centroid proximity
    # ------------------------------------------------------------------
    gdf = _dedup_by_centroid(gdf, tolerance_m=5.0)

    # ------------------------------------------------------------------
    # 6. Assign clean building_id and centroid lat/lon
    # ------------------------------------------------------------------
    gdf = gdf.reset_index(drop=True)
    # Preserve original OSM id; create a clean sequential key
    if "building_id" in gdf.columns:
        gdf = gdf.rename(columns={"building_id": "osm_building_id"})
    gdf["building_id"] = [f"B{i:07d}" for i in range(len(gdf))]

    centroids_wgs84 = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["centroid_lon"] = centroids_wgs84.x.astype("float64")
    gdf["centroid_lat"] = centroids_wgs84.y.astype("float64")

    # ------------------------------------------------------------------
    # 7. Select output columns (keep useful OSM tags)
    # ------------------------------------------------------------------
    keep_cols = ["building_id", "geometry", "centroid_lat", "centroid_lon", "area_m2"]
    for col in ["osm_building_id", "osm_id", "building_tag", "building_use", "name"]:
        if col in gdf.columns:
            keep_cols.append(col)
    gdf_out = gdf[keep_cols].copy()

    # ------------------------------------------------------------------
    # 8. Save GeoPackage
    # ------------------------------------------------------------------
    print(f"[buildings] Writing GeoPackage → {out_path.name} ...")
    gdf_out.to_file(out_path, driver="GPKG", layer="buildings")
    print(f"  saved {len(gdf_out):,} buildings")

    # ------------------------------------------------------------------
    # 9. Load into DuckDB
    # ------------------------------------------------------------------
    print(f"[buildings] Loading into DuckDB → {db_path.name} ...")
    _load_duckdb(gdf_out, out_path, db_path)

    print("[buildings] Done.")


if __name__ == "__main__":
    main()
