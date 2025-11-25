"""
preprocess/terrain.py — Mosaic, reproject, and derive terrain derivatives from GLO-30 DEM.

Inputs:
  data/raw/dem/*.tif   (4 × GLO-30 tiles, EPSG:4326, float32, ~30 m)

Outputs (all EPSG:2100, float32):
  data/processed/dem_attica_epsg2100.tif       mosaicked DEM (m)
  data/processed/slope_attica_epsg2100.tif     slope (degrees)
  data/processed/aspect_attica_epsg2100.tif    aspect (degrees, 0=N, clockwise)
  data/processed/tpi_attica_epsg2100.tif       TPI, 300 m kernel
  data/processed/tri_attica_epsg2100.tif       TRI, 300 m kernel
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Helpers — raster I/O
# ---------------------------------------------------------------------------

def _write_tif(path: Path, array: np.ndarray, profile: dict, nodata: float = -9999.0) -> None:
    """Write a single-band float32 GeoTIFF."""
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=nodata, compress="deflate", predictor=2)
    # Replace existing nodata with fill value before saving
    out = array.astype("float32")
    if profile.get("nodata") is not None:
        out = np.where(array == profile["nodata"], nodata, out)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out[np.newaxis, ...])
    print(f"  wrote {path.name}  {out.shape}  min={out[out != nodata].min():.2f}  "
          f"max={out[out != nodata].max():.2f}")


# ---------------------------------------------------------------------------
# Helpers — terrain derivatives
# ---------------------------------------------------------------------------

def _slope_aspect(dem: np.ndarray, cell_size_m: float, nodata: float = -9999.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees) using Horn's method."""
    valid = dem != nodata

    # Pad with edge replication so gradients work at boundaries
    padded = np.pad(dem, 1, mode="edge")

    # Central differences via Horn's method
    # z columns: W=0, C=1, E=2  |  rows: N=0, C=1, S=2
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cell_size_m)

    dz_dy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * cell_size_m)

    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Aspect: 0° = North, clockwise
    # atan2 convention: 0=E, ccw; transform to geographic convention
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (450.0 - aspect_deg) % 360.0  # convert to 0°N clockwise

    # Mask nodata
    slope  = np.where(valid, slope,  nodata).astype("float32")
    aspect = np.where(valid, aspect_deg, nodata).astype("float32")

    return slope, aspect


def _focal_stats(dem: np.ndarray, kernel_px: int, nodata: float = -9999.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (focal_mean, focal_mean_of_squares) using uniform square kernel."""
    # Replace nodata with nan for masked stats
    arr = dem.astype("float64")
    arr[arr == nodata] = np.nan

    # Use uniform_filter; NaN propagation handled by pre-filling with mean
    # For simplicity: fill NaN with local mean before filtering
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, np.nanmean(arr), arr)

    size = 2 * kernel_px + 1  # diameter in pixels
    focal_mean    = uniform_filter(arr_filled,    size=size, mode="reflect")
    focal_mean_sq = uniform_filter(arr_filled**2, size=size, mode="reflect")

    return focal_mean, focal_mean_sq


def _tpi(dem: np.ndarray, kernel_px: int, nodata: float = -9999.0) -> np.ndarray:
    """Topographic Position Index = elevation − focal mean."""
    focal_mean, _ = _focal_stats(dem, kernel_px, nodata)
    valid = dem != nodata
    tpi = np.where(valid, dem.astype("float64") - focal_mean, nodata)
    return tpi.astype("float32")


def _tri(dem: np.ndarray, kernel_px: int, nodata: float = -9999.0) -> np.ndarray:
    """Terrain Ruggedness Index = sqrt(mean squared deviation from focal mean).

    Equivalent to the standard deviation within the kernel, but matches the
    Riley et al. (1999) spirit extended to a wider radius.
    """
    focal_mean, focal_mean_sq = _focal_stats(dem, kernel_px, nodata)
    valid = dem != nodata
    z = dem.astype("float64")
    # Var(z_i) in kernel = E[z²] - E[z]²; then TRI = sqrt(E[(z-z_c)²]) ≈ focal std
    # Here: TRI(p) = sqrt( focal_mean_sq - 2*z_p*focal_mean + z_p² )
    tri_sq = np.maximum(0.0, focal_mean_sq - 2.0 * z * focal_mean + z**2)
    tri = np.where(valid, np.sqrt(tri_sq), nodata)
    return tri.astype("float32")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")

    dem_dir   = root / "data/raw/dem"
    proc_dir  = root / "data/processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    crs_work  = CRS.from_epsg(2100)
    feat_cfg  = cfg["features"]["terrain"]
    res_m     = 30.0   # target resolution in metres

    kernel_radius_m  = feat_cfg["features"]["tpi_300m"]["kernel_radius_m"]   # 300
    kernel_px        = round(kernel_radius_m / res_m)             # 10 pixels
    nodata_out       = -9999.0

    # ------------------------------------------------------------------
    # 1. Mosaic 4 DEM tiles
    # ------------------------------------------------------------------
    tile_paths = sorted(dem_dir.glob("*.tif"))
    print(f"[terrain] Mosaicking {len(tile_paths)} DEM tiles ...")
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, mosaic_transform = merge(datasets)
    mosaic_profile = datasets[0].profile.copy()
    for ds in datasets:
        ds.close()

    # mosaic shape: (1, rows, cols)
    dem_4326 = mosaic[0]
    mosaic_profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        count=1,
        nodata=nodata_out,
    )
    print(f"  mosaic shape: {dem_4326.shape}  CRS: EPSG:4326")

    # ------------------------------------------------------------------
    # 2. Reproject to EPSG:2100 at 30 m
    # ------------------------------------------------------------------
    print(f"[terrain] Reprojecting mosaic to EPSG:2100 at {res_m} m ...")

    # Clip bbox in source CRS for transform calculation
    dst_transform, dst_width, dst_height = calculate_default_transform(
        datasets[0].crs if False else "EPSG:4326",
        crs_work,
        mosaic_profile["width"],
        mosaic_profile["height"],
        *rasterio.transform.array_bounds(
            mosaic_profile["height"], mosaic_profile["width"], mosaic_transform
        ),
        resolution=res_m,
    )

    dem_2100 = np.empty((dst_height, dst_width), dtype="float32")
    reproject(
        source=dem_4326.astype("float32"),
        destination=dem_2100,
        src_transform=mosaic_transform,
        src_crs="EPSG:4326",
        src_nodata=None,          # GLO-30 has no nodata value set
        dst_transform=dst_transform,
        dst_crs=crs_work,
        dst_nodata=nodata_out,
        resampling=Resampling.bilinear,
    )

    dem_profile = {
        "driver": "GTiff",
        "crs": crs_work,
        "transform": dst_transform,
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "nodata": nodata_out,
        "compress": "deflate",
        "predictor": 2,
    }

    dem_path = proc_dir / "dem_attica_epsg2100.tif"
    with rasterio.open(dem_path, "w", **dem_profile) as dst:
        dst.write(dem_2100[np.newaxis, ...])
    print(f"  wrote dem_attica_epsg2100.tif  shape={dem_2100.shape}")

    # ------------------------------------------------------------------
    # 3. Derive terrain products
    # ------------------------------------------------------------------
    print(f"[terrain] Computing slope & aspect (cell_size={res_m} m) ...")
    slope, aspect = _slope_aspect(dem_2100, cell_size_m=res_m, nodata=nodata_out)
    _write_tif(proc_dir / "slope_attica_epsg2100.tif",  slope,  dem_profile, nodata_out)
    _write_tif(proc_dir / "aspect_attica_epsg2100.tif", aspect, dem_profile, nodata_out)

    print(f"[terrain] Computing TPI (radius={kernel_radius_m} m = {kernel_px} px) ...")
    tpi = _tpi(dem_2100, kernel_px, nodata=nodata_out)
    _write_tif(proc_dir / "tpi_attica_epsg2100.tif", tpi, dem_profile, nodata_out)

    print(f"[terrain] Computing TRI (radius={kernel_radius_m} m = {kernel_px} px) ...")
    tri = _tri(dem_2100, kernel_px, nodata=nodata_out)
    _write_tif(proc_dir / "tri_attica_epsg2100.tif", tri, dem_profile, nodata_out)

    # ------------------------------------------------------------------
    # 4. QC summary
    # ------------------------------------------------------------------
    valid_mask = dem_2100 != nodata_out
    print("\n[terrain] QC summary:")
    print(f"  DEM valid pixels : {valid_mask.sum():,} / {valid_mask.size:,}")
    print(f"  Elevation range  : {dem_2100[valid_mask].min():.1f} – {dem_2100[valid_mask].max():.1f} m")
    print(f"  Slope range      : {slope[slope != nodata_out].min():.1f} – {slope[slope != nodata_out].max():.1f}°")
    print(f"  TPI  range       : {tpi[tpi != nodata_out].min():.1f} – {tpi[tpi != nodata_out].max():.1f}")
    print(f"  TRI  range       : {tri[tri != nodata_out].min():.1f} – {tri[tri != nodata_out].max():.1f}")

    print("\n[terrain] Done.")


if __name__ == "__main__":
    main()
