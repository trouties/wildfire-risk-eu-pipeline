"""
features/fire_history.py — Fire-history features from EFFIS perimeters and FIRMS hotspots.

Data sources:
  data/processed/effis_perimeters_attica_epsg2100.gpkg   16 proxy perimeters, 2000-2022
  data/processed/firms_viirs_attica_epsg2100.gpkg        6130 VIIRS hotspots, 2015-2024
  data/processed/buildings_attica_epsg2100.gpkg          84,767 building polygons

Output — DuckDB table features_fire_history (84,767 rows):
  building_id TEXT
  dist_to_nearest_fire_m DOUBLE   -- dist from centroid to nearest EFFIS polygon (0 if inside)
  fire_count_5km  INTEGER         -- EFFIS fires with polygon within 5km of centroid
  fire_count_10km INTEGER         -- EFFIS fires with polygon within 10km of centroid
  ever_burned     INTEGER         -- 1 if centroid within any EFFIS polygon, else 0
  firms_hotspot_count_5km INTEGER -- FIRMS VIIRS hotspots within 5km of centroid
  recency_score   DOUBLE          -- sum of 1/(ref_year - fire_year + 1) for fires within 10km

Spatial join logic:
  - All distances are EPSG:2100 Euclidean (metric CRS), centroid-to-polygon.
  - geopandas GeoSeries.distance(polygon) returns 0 when point is inside polygon.
  - fire_count thresholds (5km, 10km) applied to the same distance used for dist_to_nearest.
  - FIRMS density uses scipy.spatial.cKDTree for efficient radius queries.
  - recency_score reference year: pipeline config date_end year (2024).

Caveats:
  - EFFIS perimeters are proxy circular approximations (proxy circular approximations of actual fire boundaries).
    Geometric containment checks and distances are approximate.
  - ever_burned = (dist_to_nearest_fire_m == 0), but re-derived via spatial containment
    for clarity.  Small floating-point differences (< 1mm) are masked.
  - Mati 2018 fire is present in EFFIS data and contributes to all fire-history features.
  - fire_history_cutoff dates in config/validation.yaml are defined but not yet
    applied as temporal filters.  The v2 LOEO evaluation excludes fire history
    features to avoid temporal leakage (see event_model.LOEO_FEATURES).
    Full per-event temporal cutoff implementation is deferred to v3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Distance matrix: buildings × EFFIS perimeters
# ---------------------------------------------------------------------------

def _build_distance_matrix(centroids_gs: gpd.GeoSeries,
                            effis_gdf: gpd.GeoDataFrame,
                            use_boundary: bool = False) -> np.ndarray:
    """Return (n_buildings, n_fires) float64 array of centroid-to-perimeter distances.

    use_boundary=False (default):
        Distance to the polygon itself.  Returns 0 for points inside — used for
        fire_count thresholds and recency weighting (consistent with 'perimeter
        within X km' semantics in features.yaml).

    use_boundary=True:
        Distance to the polygon's exterior ring (boundary linestring).  Returns a
        positive value even for points inside the polygon — used for
        dist_to_nearest_fire_m (features.yaml: 'Distance to nearest historical
        EFFIS fire perimeter boundary').  Gives meaningful discrimination even
        when all centroids fall inside proxy circular perimeters.
    """
    n_bldg = len(centroids_gs)
    n_fire = len(effis_gdf)
    dist_mat = np.empty((n_bldg, n_fire), dtype=np.float64)
    geoms = (effis_gdf.geometry.exterior if use_boundary
             else effis_gdf.geometry)
    for j, geom in enumerate(geoms):
        dist_mat[:, j] = centroids_gs.distance(geom).values
    return dist_mat


# ---------------------------------------------------------------------------
# FIRMS density via KDTree
# ---------------------------------------------------------------------------

def _firms_count_within(firms_gdf: gpd.GeoDataFrame, bldg_x: np.ndarray,
                         bldg_y: np.ndarray, radius_m: float) -> np.ndarray:
    """Count FIRMS hotspots within radius_m of each building centroid."""
    firms_x = firms_gdf.geometry.x.values
    firms_y = firms_gdf.geometry.y.values
    firms_coords = np.column_stack([firms_x, firms_y])
    bldg_coords  = np.column_stack([bldg_x,  bldg_y])

    tree = cKDTree(firms_coords)
    # workers=-1 enables parallel processing on multicore machines
    indices = tree.query_ball_point(bldg_coords, r=radius_m, workers=-1)
    return np.array([len(idx) for idx in indices], dtype=np.int32)


# ---------------------------------------------------------------------------
# Recency score
# ---------------------------------------------------------------------------

def _recency_score(dist_mat: np.ndarray, fire_years: np.ndarray,
                   ref_year: int, max_dist_m: float = 10_000.0) -> np.ndarray:
    """Recency-weighted fire count within max_dist_m.

    weight_j = 1 / (ref_year - year_j + 1)
    score_i  = sum_j [ weight_j  if  dist_mat[i,j] < max_dist_m ]

    Higher scores for buildings near recent fires.
    Fires beyond max_dist_m contribute 0.
    """
    weights = 1.0 / (ref_year - fire_years.astype(float) + 1.0)
    within  = dist_mat < max_dist_m          # (n_bldg, n_fire) bool
    scores  = within.dot(weights)            # (n_bldg,)
    return scores.astype(np.float64)


# ---------------------------------------------------------------------------
# DuckDB write
# ---------------------------------------------------------------------------

def _write_duckdb(df: pd.DataFrame, db_path: Path) -> int:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("DROP TABLE IF EXISTS features_fire_history;")
    con.register("_fh_df", df)
    con.execute("""
        CREATE TABLE features_fire_history AS
        SELECT
            building_id,
            dist_to_nearest_fire_m,
            fire_count_5km,
            fire_count_10km,
            ever_burned,
            firms_hotspot_count_5km,
            recency_score
        FROM _fh_df
    """)
    n = con.execute("SELECT count(*) FROM features_fire_history").fetchone()[0]
    row = con.execute("""
        SELECT
            round(avg(dist_to_nearest_fire_m), 0)            AS avg_dist,
            round(avg(fire_count_5km),  2)                   AS avg_fc5,
            round(avg(fire_count_10km), 2)                   AS avg_fc10,
            sum(ever_burned)                                  AS n_burned,
            round(avg(firms_hotspot_count_5km), 2)           AS avg_firms,
            round(avg(recency_score), 4)                      AS avg_rec
        FROM features_fire_history
    """).fetchone()
    print("  [duckdb] features_fire_history:")
    print(f"    rows={n:,}  avg_dist_to_fire={row[0]:.0f}m")
    print(f"    avg_fire_count_5km={row[1]}  avg_fire_count_10km={row[2]}")
    print(f"    ever_burned={int(row[3]):,}  avg_firms_5km={row[4]}  avg_recency={row[5]}")
    con.close()
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")
    proc = root / "data/processed"
    db_path  = root / cfg["pipeline"]["paths"]["db"]
    ref_year = int(cfg["pipeline"]["run"]["date_end"].split("-")[0])  # 2024

    # ------------------------------------------------------------------
    # 1. Load buildings (centroids in EPSG:2100)
    # ------------------------------------------------------------------
    print("[features/fire_history] Loading buildings ...")
    gdf_bldg = gpd.read_file(proc / "buildings_attica_epsg2100.gpkg")
    centroids  = gdf_bldg.geometry.centroid
    cx = centroids.x.values
    cy = centroids.y.values
    ids = gdf_bldg["building_id"].values
    n   = len(gdf_bldg)
    # GeoSeries of centroid Points for geopandas distance ops
    from shapely.geometry import Point
    centroids_gs = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(cx, cy)], crs="EPSG:2100"
    )
    print(f"  {n:,} buildings loaded")

    # ------------------------------------------------------------------
    # 2. Load EFFIS perimeters and FIRMS hotspots
    # ------------------------------------------------------------------
    print("[features/fire_history] Loading EFFIS perimeters ...")
    gdf_effis = gpd.read_file(proc / "effis_perimeters_attica_epsg2100.gpkg")
    fire_years = gdf_effis["year"].astype(int).values
    print(f"  {len(gdf_effis)} perimeters  years={sorted(fire_years.tolist())}")

    print("[features/fire_history] Loading FIRMS hotspots ...")
    gdf_firms = gpd.read_file(proc / "firms_viirs_attica_epsg2100.gpkg")
    print(f"  {len(gdf_firms):,} hotspots  "
          f"dates={gdf_firms['acq_date'].min()} – {gdf_firms['acq_date'].max()}")

    # ------------------------------------------------------------------
    # 3a. Polygon distance matrix — used for fire_count thresholds and recency.
    #     Returns 0 for buildings inside a perimeter (consistent with
    #     "perimeter within X km" semantics).
    # ------------------------------------------------------------------
    print(f"[features/fire_history] Polygon distances ({n:,} × {len(gdf_effis)}) ...")
    dist_mat_poly = _build_distance_matrix(centroids_gs, gdf_effis, use_boundary=False)

    # ------------------------------------------------------------------
    # 3b. Boundary distance matrix — used for dist_to_nearest_fire_m.
    #     Returns positive values even for centroids inside perimeters,
    #     giving meaningful discrimination where all buildings are enclosed
    #     by large proxy-circular EFFIS perimeters.
    # ------------------------------------------------------------------
    print(f"[features/fire_history] Boundary distances ({n:,} × {len(gdf_effis)}) ...")
    dist_mat_bnd = _build_distance_matrix(centroids_gs, gdf_effis, use_boundary=True)

    # ------------------------------------------------------------------
    # 4. Derive EFFIS-based features
    # ------------------------------------------------------------------
    dist_to_nearest = dist_mat_bnd.min(axis=1)   # distance to nearest exterior ring

    fire_count_5km  = (dist_mat_poly < 5_000.0).sum(axis=1).astype(np.int32)
    fire_count_10km = (dist_mat_poly < 10_000.0).sum(axis=1).astype(np.int32)

    # ever_burned: centroid inside any perimeter polygon.
    # NOTE: With EFFIS proxy circular approximations, this is often 100% within
    # the dataset AOI (all buildings appear inside at least one proxy circle).
    # Retain the column but flag the degeneracy in output and caveats.
    all_perimeters_union = gdf_effis.geometry.unary_union
    ever_burned = centroids_gs.within(all_perimeters_union).astype(np.int32).values

    pct_burned = ever_burned.mean() * 100
    degenerate_note = " ⚠ DEGENERATE (EFFIS proxy circles cover 100% of AOI)" if pct_burned > 95 else ""
    print(f"  dist_to_nearest (boundary): mean={dist_to_nearest.mean():.0f}m  "
          f"min={dist_to_nearest.min():.0f}m  max={dist_to_nearest.max():.0f}m")
    print(f"  fire_count_5km: mean={fire_count_5km.mean():.2f}  "
          f"max={fire_count_5km.max()}")
    print(f"  fire_count_10km: mean={fire_count_10km.mean():.2f}  "
          f"max={fire_count_10km.max()}")
    print(f"  ever_burned: {ever_burned.sum():,} ({pct_burned:.2f}%){degenerate_note}")

    # ------------------------------------------------------------------
    # 5. FIRMS hotspot count within 5km
    # ------------------------------------------------------------------
    print("[features/fire_history] FIRMS hotspot count within 5km ...")
    firms_count = _firms_count_within(gdf_firms, cx, cy, radius_m=5_000.0)
    print(f"  mean={firms_count.mean():.2f}  max={firms_count.max():,}  "
          f"buildings_with_hotspot={(firms_count > 0).sum():,}")

    # ------------------------------------------------------------------
    # 6. Recency score (fires within 10km, weighted by 1/(ref_year - year + 1))
    # ------------------------------------------------------------------
    print(f"[features/fire_history] Recency score (ref_year={ref_year}, radius=10km) ...")
    rec_score = _recency_score(dist_mat_poly, fire_years, ref_year, max_dist_m=10_000.0)
    print(f"  mean={rec_score.mean():.4f}  max={rec_score.max():.4f}  "
          f"buildings_with_score={(rec_score > 0).sum():,}")

    # ------------------------------------------------------------------
    # 7. Pre-write sanity checks
    # ------------------------------------------------------------------
    assert len(ids) == n
    assert dist_to_nearest.min() >= 0, "Negative distance"
    assert fire_count_5km.max()  <= len(gdf_effis), "fire_count_5km > n_perimeters"
    assert fire_count_10km.max() <= len(gdf_effis), "fire_count_10km > n_perimeters"
    assert np.isin(ever_burned, [0, 1]).all(), "ever_burned not in {0,1}"
    assert firms_count.min()  >= 0, "Negative FIRMS count"
    assert rec_score.min()    >= 0, "Negative recency score"
    print("[features/fire_history] Pre-write checks passed.")

    # ------------------------------------------------------------------
    # 8. Build DataFrame and write to DuckDB
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "building_id":            ids,
        "dist_to_nearest_fire_m": dist_to_nearest,
        "fire_count_5km":         fire_count_5km,
        "fire_count_10km":        fire_count_10km,
        "ever_burned":            ever_burned,
        "firms_hotspot_count_5km": firms_count,
        "recency_score":          rec_score,
    })

    print(f"[features/fire_history] Writing → {db_path.name} ...")
    count = _write_duckdb(df, db_path)
    assert count == n, f"DuckDB row count {count} != expected {n}"
    print(f"[features/fire_history] Done — {count:,} rows in features_fire_history.")


if __name__ == "__main__":
    main()
