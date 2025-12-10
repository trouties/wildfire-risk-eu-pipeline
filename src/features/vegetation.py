"""
features/vegetation.py — Vegetation and WUI features from WorldCover + CORINE binary.

Data sources:
  data/processed/corine_binary_epsg2100.tif     10m uint8 {0=non-veg, 1=veg, 255=nodata}
  data/raw/corine/landcover_attica_2021.tif      ESA WorldCover 2021, EPSG:4326, ~10m
  data/processed/buildings_attica_epsg2100.gpkg  84,767 building polygons, EPSG:2100

WorldCover class mapping (proxy for CORINE CLC):
  class 10 → Tree cover   → proxy for CORINE 311-324 (forest)
  class 20 → Shrubland    → proxy for CORINE 323 (sclerophyllous vegetation)
  class 30 → Grassland    → also in fire-veg binary but not tracked separately

Output — DuckDB table features_vegetation (84,767 rows):
  building_id TEXT
  ndvi_mean_100m DOUBLE        -- NULL: NDVI source not available (data gap)
  ndvi_mean_500m DOUBLE        -- NULL
  ndvi_max_500m  DOUBLE        -- NULL
  veg_fraction_100m DOUBLE     -- fraction fire-veg pixels in 100m square kernel
  veg_fraction_500m DOUBLE     -- fraction fire-veg pixels in 500m square kernel
  dist_to_forest_m  DOUBLE     -- Euclidean dist to nearest WorldCover class-10 pixel
  dist_to_scrubland_m DOUBLE   -- Euclidean dist to nearest WorldCover class-20 pixel
  wui_class INTEGER            -- 0=non-WUI, 1=interface WUI, 2=intermix WUI
  veg_continuity_500m DOUBLE   -- largest connected veg patch / total veg in 500m radius

Assumptions and thresholds:
  - veg_fraction uses square kernel (uniform_filter) as approximation of circular buffer
  - dist_to_forest_m = 0 for buildings with centroid on a tree-cover pixel
  - wui_class thresholds from config: density>=6.17/km² + veg>=0.5 → interface (1);
    density<6.17/km² + veg>=0.5 → intermix (2); else non-WUI (0)
  - veg_continuity: 0 for buildings not on vegetated pixels
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.ndimage import label as ndimage_label
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# WorldCover class codes relevant to vegetation features
# ---------------------------------------------------------------------------
WC_TREE  = 10   # Tree cover  → proxy for CORINE forest (311-324)
WC_SHRUB = 20   # Shrubland   → proxy for CORINE scrubland (323)


# ---------------------------------------------------------------------------
# Coordinate sampling helpers
# ---------------------------------------------------------------------------

def _xy_to_rowcol(transform, xs: np.ndarray, ys: np.ndarray,
                  height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert world coordinates to pixel (row, col), clamped to raster extent."""
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows, dtype=int), 0, height - 1)
    cols = np.clip(np.asarray(cols, dtype=int), 0, width - 1)
    return rows, cols


def _sample_array(array: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray,
                  nodata=None) -> np.ndarray:
    """Sample a 2D array at world coordinates; replace nodata with NaN."""
    rows, cols = _xy_to_rowcol(transform, xs, ys, array.shape[0], array.shape[1])
    vals = array[rows, cols].astype(float)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    return vals


# ---------------------------------------------------------------------------
# Vegetation fraction (focal mean)
# ---------------------------------------------------------------------------

def _compute_veg_fraction(binary: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray,
                           radius_m: float, pixel_size_m: float) -> np.ndarray:
    """Fraction of fire-veg pixels within radius_m of each centroid.

    Uses scipy.ndimage.uniform_filter (square kernel) as a fast approximation
    of the circular buffer mean. Error vs. true circular mean ≈ 1 − π/4 ≈ 21%;
    documented as acceptable for this portfolio application.

    nodata pixels (255) are treated as non-vegetated (value 0) so fractions
    near the AOI boundary are conservative (slightly underestimated).
    """
    arr = np.where(binary == 255, 0, binary).astype(float)
    radius_px = max(1, int(round(radius_m / pixel_size_m)))
    size = 2 * radius_px + 1
    focal = uniform_filter(arr, size=size, mode="reflect")
    # Clip to [0, 1]: uniform_filter can produce tiny negative values at borders
    # due to floating-point precision when the source array has nodata=0 edges.
    focal = np.clip(focal, 0.0, 1.0)
    return _sample_array(focal, transform, xs, ys)


# ---------------------------------------------------------------------------
# Class-specific distance transforms
# ---------------------------------------------------------------------------

def _reproject_class_mask(raw_path: Path, class_code: int,
                          dst_transform, dst_shape: tuple[int, int],
                          dst_crs) -> np.ndarray:
    """Load raw WorldCover, extract class_code as binary mask, reproject to target grid.

    Returns uint8 array: 1 where class_code, 0 elsewhere.
    Uses nearest-neighbour resampling (categorical data).
    """
    with rasterio.open(raw_path) as src:
        raw = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    class_mask = (raw == class_code).astype(np.uint8)
    dst_mask = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=class_mask,
        destination=dst_mask,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst_mask


def _class_distance_m(class_mask_uint8: np.ndarray, res_y: float, res_x: float) -> np.ndarray:
    """Euclidean distance in metres to nearest pixel of the target class.

    distance_transform_edt convention:
      input 0 = target (feature)  → output 0
      input 1 = background        → output = distance to nearest target pixel

    With sampling=(res_y, res_x) in metres the output is directly in metres.
    """
    # Invert: 0 where class exists (feature), 1 elsewhere
    edt_input = (~class_mask_uint8.astype(bool)).astype(np.uint8)
    return distance_transform_edt(edt_input, sampling=(res_y, res_x)).astype(np.float32)


# ---------------------------------------------------------------------------
# Building density
# ---------------------------------------------------------------------------

def _building_density_per_km2(xs: np.ndarray, ys: np.ndarray,
                               radius_m: float = 500.0) -> np.ndarray:
    """Count of other buildings within radius_m of each building, per km²."""
    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)
    # query_ball_point returns all points within radius, including self
    indices = tree.query_ball_point(coords, r=radius_m, workers=-1)
    counts = np.array([len(idx) - 1 for idx in indices], dtype=float)  # exclude self
    area_km2 = np.pi * (radius_m / 1_000.0) ** 2
    return counts / area_km2


# ---------------------------------------------------------------------------
# WUI classification
# ---------------------------------------------------------------------------

def _wui_class(veg_frac_500m: np.ndarray, density_per_km2: np.ndarray,
               density_thresh: float, veg_thresh: float) -> np.ndarray:
    """
    0 = non-WUI        (veg_fraction_500m < veg_thresh)
    1 = interface WUI  (density >= density_thresh AND veg_fraction >= veg_thresh)
    2 = intermix WUI   (density <  density_thresh AND veg_fraction >= veg_thresh)
    Thresholds from config/features.yaml → terrain.features.wui_class.
    """
    wui = np.zeros(len(veg_frac_500m), dtype=np.int32)
    high_veg = veg_frac_500m >= veg_thresh
    wui[high_veg & (density_per_km2 >= density_thresh)] = 1
    wui[high_veg & (density_per_km2 <  density_thresh)] = 2
    return wui


# ---------------------------------------------------------------------------
# Vegetation continuity
# ---------------------------------------------------------------------------

def _veg_continuity_500m(binary: np.ndarray, transform,
                          xs: np.ndarray, ys: np.ndarray,
                          pixel_size_m: float, radius_m: float = 500.0) -> np.ndarray:
    """Fraction of vegetated pixels in radius_m belonging to the same connected patch
    as the building centroid pixel.

    Algorithm:
      1. scipy.ndimage.label on binary == 1 (once, full raster)
      2. For each centroid: find center label; if non-vegetated → 0;
         else extract circular window, count pixels with same label / total veg pixels.

    Returns 0 for buildings whose centroid pixel is not vegetated.
    """
    veg_mask = binary == 1
    print("  [continuity] Running connected-component labeling ...")
    labeled, n_labels = ndimage_label(veg_mask)
    print(f"  [continuity] {n_labels:,} connected vegetated patches found")

    height, width = binary.shape
    ctr_rows, ctr_cols = _xy_to_rowcol(transform, xs, ys, height, width)

    radius_px = max(1, int(round(radius_m / pixel_size_m)))
    # Precompute circular boolean mask for the window
    ry, rx = np.ogrid[-radius_px: radius_px + 1, -radius_px: radius_px + 1]
    circ = (ry ** 2 + rx ** 2) <= radius_px ** 2

    results = np.zeros(len(xs), dtype=np.float32)

    for i, (r, c) in enumerate(zip(ctr_rows, ctr_cols)):
        center_label = labeled[r, c]
        if center_label == 0:          # centroid not on vegetated pixel
            results[i] = 0.0
            continue

        # Window bounds — clamped to raster edges
        r0, r1 = max(0, r - radius_px), min(height, r + radius_px + 1)
        c0, c1 = max(0, c - radius_px), min(width,  c + radius_px + 1)

        # Adjust circular mask for edge-clamped windows
        mr0 = radius_px - (r - r0)
        mr1 = mr0 + (r1 - r0)
        mc0 = radius_px - (c - c0)
        mc1 = mc0 + (c1 - c0)
        local_circ = circ[mr0:mr1, mc0:mc1]

        win_label = labeled[r0:r1, c0:c1]
        win_veg   = veg_mask[r0:r1, c0:c1]

        total_veg = int((win_veg & local_circ).sum())
        if total_veg == 0:
            results[i] = 0.0
            continue

        same_patch = (win_label == center_label) & local_circ
        results[i] = float(same_patch.sum()) / total_veg

    return results


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
    con.execute("DROP TABLE IF EXISTS features_vegetation;")
    con.register("_veg_df", df)
    con.execute("""
        CREATE TABLE features_vegetation AS
        SELECT
            building_id,
            ndvi_mean_100m,
            ndvi_mean_500m,
            ndvi_max_500m,
            veg_fraction_100m,
            veg_fraction_500m,
            dist_to_forest_m,
            dist_to_scrubland_m,
            wui_class,
            veg_continuity_500m
        FROM _veg_df
    """)
    n = con.execute("SELECT count(*) FROM features_vegetation").fetchone()[0]
    row = con.execute("""
        SELECT
            round(avg(veg_fraction_100m), 3)                   AS avg_vf100,
            round(avg(veg_fraction_500m), 3)                   AS avg_vf500,
            round(avg(dist_to_forest_m),  1)                   AS avg_df,
            round(avg(dist_to_scrubland_m), 1)                 AS avg_ds,
            count(*) FILTER (WHERE wui_class = 0)              AS wui0,
            count(*) FILTER (WHERE wui_class = 1)              AS wui1,
            count(*) FILTER (WHERE wui_class = 2)              AS wui2,
            round(avg(veg_continuity_500m), 3)                 AS avg_cont,
            count(*) FILTER (WHERE ndvi_mean_100m IS NULL)     AS null_ndvi
        FROM features_vegetation
    """).fetchone()
    print("  [duckdb] features_vegetation:")
    print(f"    rows={n:,}  avg_vf100={row[0]}  avg_vf500={row[1]}")
    print(f"    avg_dist_forest={row[2]}m  avg_dist_shrub={row[3]}m")
    print(f"    wui: 0={row[4]:,}  1={row[5]:,}  2={row[6]:,}")
    print(f"    avg_continuity={row[7]}  null_ndvi={row[8]:,} (all expected NULL)")
    con.close()
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")
    proc = root / "data/processed"
    db_path   = root / cfg["pipeline"]["paths"]["db"]
    raw_wc    = root / "data/raw/corine/landcover_attica_2021.tif"
    feat_cfg  = cfg["features"]["vegetation"]["features"]

    # ------------------------------------------------------------------
    # 1. Load CORINE binary raster
    # ------------------------------------------------------------------
    print("[features/vegetation] Loading CORINE binary ...")
    with rasterio.open(proc / "corine_binary_epsg2100.tif") as src:
        binary      = src.read(1)
        transform   = src.transform
        crs         = src.crs
        res_y, res_x = src.res   # (10.0, 10.0)
    pixel_size_m = res_y
    print(f"  shape={binary.shape}  res={pixel_size_m}m")

    # ------------------------------------------------------------------
    # 2. Load buildings (centroids in EPSG:2100)
    # ------------------------------------------------------------------
    print("[features/vegetation] Loading buildings ...")
    gdf = gpd.read_file(proc / "buildings_attica_epsg2100.gpkg")
    cx  = gdf.geometry.centroid.x.values
    cy  = gdf.geometry.centroid.y.values
    ids = gdf["building_id"].values
    n   = len(gdf)
    print(f"  {n:,} buildings")

    # ------------------------------------------------------------------
    # 3. Vegetation fractions (100m and 500m)
    # ------------------------------------------------------------------
    print("[features/vegetation] veg_fraction_100m (100m square kernel) ...")
    vf100 = _compute_veg_fraction(binary, transform, cx, cy, 100.0, pixel_size_m)
    print(f"  mean={vf100.mean():.3f}  min={vf100.min():.3f}  max={vf100.max():.3f}")

    print("[features/vegetation] veg_fraction_500m (500m square kernel) ...")
    vf500 = _compute_veg_fraction(binary, transform, cx, cy, 500.0, pixel_size_m)
    print(f"  mean={vf500.mean():.3f}  min={vf500.min():.3f}  max={vf500.max():.3f}")

    # ------------------------------------------------------------------
    # 4. Distance to forest (WorldCover class 10 = Tree cover)
    # ------------------------------------------------------------------
    print("[features/vegetation] dist_to_forest_m (WorldCover class 10 → Tree cover) ...")
    tree_mask = _reproject_class_mask(raw_wc, WC_TREE, transform, binary.shape, crs)
    print(f"  tree-cover pixels: {tree_mask.sum():,} / {tree_mask.size:,}")
    dist_forest_arr = _class_distance_m(tree_mask, res_y, res_x)
    dist_forest = _sample_array(dist_forest_arr, transform, cx, cy)
    print(f"  mean={dist_forest.mean():.0f}m  max={dist_forest.max():.0f}m  "
          f"buildings_on_forest={(tree_mask[*_xy_to_rowcol(transform, cx, cy, *binary.shape)]).sum():,}")
    del tree_mask, dist_forest_arr

    # ------------------------------------------------------------------
    # 5. Distance to scrubland (WorldCover class 20 = Shrubland)
    # ------------------------------------------------------------------
    print("[features/vegetation] dist_to_scrubland_m (WorldCover class 20 → Shrubland) ...")
    shrub_mask = _reproject_class_mask(raw_wc, WC_SHRUB, transform, binary.shape, crs)
    print(f"  shrubland pixels: {shrub_mask.sum():,} / {shrub_mask.size:,}")
    dist_shrub_arr = _class_distance_m(shrub_mask, res_y, res_x)
    dist_shrub = _sample_array(dist_shrub_arr, transform, cx, cy)
    print(f"  mean={dist_shrub.mean():.0f}m  max={dist_shrub.max():.0f}m")
    del shrub_mask, dist_shrub_arr

    # ------------------------------------------------------------------
    # 6. Building density → WUI class
    # ------------------------------------------------------------------
    print("[features/vegetation] Building density (500m radius) for WUI classification ...")
    density = _building_density_per_km2(cx, cy, radius_m=500.0)
    print(f"  density: mean={density.mean():.1f}/km²  p50={np.median(density):.1f}/km²  "
          f"max={density.max():.1f}/km²")

    d_thresh = feat_cfg["wui_class"]["interface_density_threshold"]  # 6.17
    v_thresh = feat_cfg["wui_class"]["veg_fraction_threshold"]       # 0.50
    wui = _wui_class(vf500, density, d_thresh, v_thresh)
    wui_u, wui_c = np.unique(wui, return_counts=True)
    print("  WUI classes: " + "  ".join(f"{u}={c:,}" for u, c in zip(wui_u, wui_c)))

    # ------------------------------------------------------------------
    # 7. Vegetation continuity 500m
    # ------------------------------------------------------------------
    print("[features/vegetation] veg_continuity_500m (may take ~1 min) ...")
    veg_cont = _veg_continuity_500m(binary, transform, cx, cy, pixel_size_m, radius_m=500.0)
    n_in_veg = int((veg_cont > 0).sum())
    print(f"  mean={veg_cont.mean():.3f}  buildings_on_veg={n_in_veg:,} / {n:,}")

    # ------------------------------------------------------------------
    # 8. NDVI columns → NULL (data gap)
    # ------------------------------------------------------------------
    # ndvi_season_composite_epsg2100.tif was not produced in T01-T07.
    # Multi-temporal Sentinel-2 or Landsat imagery was not acquired.
    # Columns are included with NULL values so the schema matches features.yaml
    # and T10 normalization can handle them with imputation.
    null_col = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # 9. Pre-write sanity checks
    # ------------------------------------------------------------------
    assert len(ids) == n
    assert np.all((vf100 >= 0) & (vf100 <= 1)), "veg_fraction_100m out of [0,1]"
    assert np.all((vf500 >= 0) & (vf500 <= 1)), "veg_fraction_500m out of [0,1]"
    assert np.all(dist_forest >= 0), "dist_to_forest_m negative"
    assert np.all(dist_shrub  >= 0), "dist_to_scrubland_m negative"
    assert np.all(np.isin(wui, [0, 1, 2])), "wui_class out of {0,1,2}"
    assert np.all((veg_cont >= 0) & (veg_cont <= 1)), "veg_continuity_500m out of [0,1]"
    print("[features/vegetation] Pre-write checks passed.")

    # ------------------------------------------------------------------
    # 10. Build DataFrame and write to DuckDB
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "building_id":         ids,
        "ndvi_mean_100m":      null_col,
        "ndvi_mean_500m":      null_col,
        "ndvi_max_500m":       null_col,
        "veg_fraction_100m":   vf100,
        "veg_fraction_500m":   vf500,
        "dist_to_forest_m":    dist_forest,
        "dist_to_scrubland_m": dist_shrub,
        "wui_class":           wui,
        "veg_continuity_500m": veg_cont,
    })

    print(f"[features/vegetation] Writing → {db_path.name} ...")
    count = _write_duckdb(df, db_path)
    assert count == n, f"DuckDB row count {count} != expected {n}"
    print(f"[features/vegetation] Done — {count:,} rows in features_vegetation.")


if __name__ == "__main__":
    main()
