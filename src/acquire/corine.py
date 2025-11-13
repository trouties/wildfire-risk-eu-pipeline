"""
src/acquire/corine.py
---------------------
Download land cover data clipped to the Attica AOI.

Primary source: ESA WorldCover 2021 v200 (10 m resolution) from AWS Open Data.
  - Tiles are publicly accessible Cloud Optimized GeoTIFFs on S3.
  - No authentication required.
  - URL pattern: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/
                 ESA_WorldCover_10m_2021_v200_{TILE}_Map.tif
  - Tiles are 3° × 3° (SW-corner naming, EPSG:4326).
  - Attica AOI [23.4, 37.6, 24.2, 38.3] requires two tiles:
      N36E021 (lat 36-39, lon 21-24)  ~63 MB
      N36E024 (lat 36-39, lon 24-27)  ~19 MB

Previous source: CORINE Land Cover 2018 via EEA WMS.
  Status: EEA WMS GetMap consistently returns blank (all-255) tiles for this
  AOI. The ArcGIS ImageServer ExportImage endpoint returns 404 (service not
  found). Manual download from land.copernicus.eu requires authentication.
  Decision: switch to WorldCover as automated primary source.

WorldCover 2021 classification (uint8 pixel values)
----------------------------------------------------
Code | Class               | Fire risk relevance
-----|---------------------|--------------------
  10 | Tree cover          | HIGH  (crown fire, ember transport)
  20 | Shrubland           | HIGH  (fast spread, maquis/garrigue in Med)
  30 | Grassland           | MED   (fine fuel, fast spread but low intensity)
  40 | Cropland            | LOW   (irrigated or harvested, low fuel load)
  50 | Built-up            | NONE  (exposure object, not fuel)
  60 | Bare/sparse veg     | LOW   (limited fuel)
  70 | Snow and ice        | NONE
  80 | Permanent water     | NONE
  90 | Herbaceous wetland  | LOW
  95 | Mangroves           | LOW
 100 | Moss and lichen     | NONE

Mapping to CORINE CLC equivalents (for portfolio documentation)
  WorldCover 10 ≈ CLC 311/312/313 (broad-leaved, coniferous, mixed forest)
  WorldCover 20 ≈ CLC 322/323/324 (moors, sclerophyllous veg, transitional shrub)
  WorldCover 30 ≈ CLC 321 (natural grasslands)
  WorldCover 60 ≈ CLC 332/333 (bare rocks, sparsely vegetated)

Output
------
data/raw/corine/
  landcover_attica_2021.tif   (10 m, uint8, EPSG:4326, clipped to AOI)
  manifest.json               (provenance record)

CRS / Resolution / Cadence notes
---------------------------------
- Output CRS: EPSG:4326 (native WorldCover CRS)
- Resolution: ~10 m (0.000083333° per pixel, native WorldCover)
  Coarser resampling handled downstream in T05 if needed.
- Vintage: 2021 (WorldCover v200 reference year)
- Temporal: static snapshot
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.windows import from_bounds

from src.utils.config import get_bbox, load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WorldCover constants
# ---------------------------------------------------------------------------

_WC_S3_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map"
)

# Fire-relevant WorldCover class codes
FIRE_VEG_CODES: frozenset[int] = frozenset({10, 20, 30})   # tree, shrub, grass
HIGH_RISK_CODES: frozenset[int] = frozenset({10, 20})       # tree + shrub (Med)
FOREST_CODES: frozenset[int] = frozenset({10})               # tree cover only

# Human-readable labels
WC_CLASS_NAMES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
    0: "No data",
}

# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def _tiles_for_bbox(bbox: list[float]) -> list[str]:
    """Return WorldCover 3° × 3° tile IDs covering the given [W, S, E, N] bbox.

    Tile names use the lower-left (SW) corner at multiples of 3°.
    """
    west, south, east, north = bbox
    import math
    lat_min = int(math.floor(south / 3) * 3)
    lat_max = int(math.floor(north / 3) * 3)
    lon_min = int(math.floor(west  / 3) * 3)
    lon_max = int(math.floor(east  / 3) * 3)

    tiles = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            lat_s = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_s = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tiles.append(f"{lat_s}{lon_s}")
            lon += 3
        lat += 3
    return tiles


def _tile_url(tile_id: str) -> str:
    """Return the HTTPS URL for a WorldCover 2021 v200 tile."""
    fname = f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
    return f"{_WC_S3_BASE}/{fname}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_tile(
    tile_id: str,
    dest_path: Path,
    force: bool = False,
) -> bool:
    """Download a single WorldCover tile to dest_path.

    Returns True on success, False on network failure.
    """
    if dest_path.exists() and not force:
        logger.info("Tile already cached: %s (%s MB)",
                    tile_id, dest_path.stat().st_size // 1024 // 1024)
        return True

    url = _tile_url(tile_id)
    logger.info("Downloading tile %s ...", tile_id)

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Tile %s download failed: %s", tile_id, exc)
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest_path.parent, suffix=".tif", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            tmp.write(chunk)

    tmp_path.rename(dest_path)
    size_mb = dest_path.stat().st_size / 1024 / 1024
    logger.info("Tile %s saved: %.1f MB → %s", tile_id, size_mb, dest_path.name)
    return True


# ---------------------------------------------------------------------------
# Mosaic + clip
# ---------------------------------------------------------------------------

def _mosaic_and_clip(
    tile_paths: list[Path],
    bbox: list[float],
    output_path: Path,
) -> None:
    """Mosaic WorldCover tiles and clip to the AOI bounding box.

    Writes a uint8 GeoTIFF in EPSG:4326, compressed with LZW.
    """
    west, south, east, north = bbox

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_transform = merge(datasets)
    finally:
        for ds in datasets:
            ds.close()

    # Crop to AOI window
    height, width = mosaic.shape[1], mosaic.shape[2]
    win = from_bounds(west, south, east, north, mosaic_transform)
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    row_end = min(height, int(win.row_off + win.height))
    col_end = min(width,  int(win.col_off + win.width))

    clipped = mosaic[:, row_off:row_end, col_off:col_end]

    from rasterio.transform import from_bounds as fb

    # Recompute transform for the clipped extent
    # Use actual pixel boundaries rather than exact bbox
    pixel_size_x = mosaic_transform.a
    pixel_size_y = abs(mosaic_transform.e)
    clip_west  = mosaic_transform.c + col_off * pixel_size_x
    clip_north = mosaic_transform.f - row_off * pixel_size_y
    clip_east  = clip_west  + clipped.shape[2] * pixel_size_x
    clip_south = clip_north - clipped.shape[1] * pixel_size_y

    out_transform = fb(clip_west, clip_south, clip_east, clip_north,
                       clipped.shape[2], clipped.shape[1])

    out_profile = {
        "driver":    "GTiff",
        "dtype":     rasterio.uint8,
        "count":     1,
        "height":    clipped.shape[1],
        "width":     clipped.shape[2],
        "crs":       CRS.from_epsg(4326),
        "transform": out_transform,
        "compress":  "lzw",
        "nodata":    0,
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(clipped[0], 1)
        dst.update_tags(
            TIFFTAG_IMAGEDESCRIPTION=(
                "ESA WorldCover 2021 v200 — land cover class codes (uint8). "
                "Source: ESA WorldCover / AWS Open Data. "
                f"AOI: {bbox}. Generated: {datetime.now(timezone.utc).isoformat()}"
            )
        )

    logger.info(
        "Mosaic+clip done: %dx%d px → %s",
        clipped.shape[2], clipped.shape[1], output_path.name,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_corine(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    """Download ESA WorldCover 2021 land cover for the Attica AOI.

    This function is named ``download_corine`` for pipeline interface
    compatibility. The underlying source is ESA WorldCover (not CORINE CLC),
    following a data source substitution documented in the module header.

    Args:
        output_dir: Target directory (created if missing).
        cfg:        Merged pipeline config.
        force:      Re-download even if output file exists.

    Returns:
        Path to the output GeoTIFF.

    Raises:
        RuntimeError: If all tile downloads fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = get_bbox(cfg)  # [W, S, E, N]

    out_tif  = output_dir / "landcover_attica_2021.tif"
    manifest = output_dir / "manifest.json"

    if out_tif.exists() and not force:
        size_mb = out_tif.stat().st_size / 1024 / 1024
        logger.info(
            "Land cover output already exists: %s (%.1f MB). "
            "Pass force=True to re-download.",
            out_tif.name, size_mb,
        )
        return out_tif

    # ── Discover tiles ─────────────────────────────────────────────────────────
    tile_ids = _tiles_for_bbox(bbox)
    logger.info("WorldCover tiles needed for AOI %s: %s", bbox, tile_ids)

    tile_cache_dir = output_dir / "_tiles"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Download tiles ─────────────────────────────────────────────────────────
    tile_paths: list[Path] = []
    for tid in tile_ids:
        dest = tile_cache_dir / f"ESA_WorldCover_10m_2021_v200_{tid}_Map.tif"
        ok = _download_tile(tid, dest, force=force)
        if not ok:
            raise RuntimeError(
                f"WorldCover tile {tid} download failed. "
                "Check network connectivity and try again."
            )
        tile_paths.append(dest)

    # ── Mosaic + clip to AOI ───────────────────────────────────────────────────
    logger.info("Mosaicking %d tiles and clipping to AOI...", len(tile_paths))
    _mosaic_and_clip(tile_paths, bbox, out_tif)

    # ── Verify ────────────────────────────────────────────────────────────────
    with rasterio.open(out_tif) as ds:
        data = ds.read(1)
        unique_codes = sorted(int(v) for v in np.unique(data) if v > 0)
        tile_info = {
            "crs":               str(ds.crs),
            "width":             ds.width,
            "height":            ds.height,
            "resolution_deg":    round(abs(ds.transform.a), 9),
            "dtype":             str(ds.dtypes[0]),
            "nodata":            ds.nodata,
            "unique_wc_codes":   unique_codes,
            "fire_veg_codes_present": sorted(
                int(c) for c in FIRE_VEG_CODES if c in unique_codes
            ),
            "non_zero_pixels":   int((data > 0).sum()),
            "size_mb":           round(out_tif.stat().st_size / 1024 / 1024, 2),
        }

    logger.info(
        "WorldCover verified: %dx%d px, %d classes, %.1f MB",
        tile_info["width"], tile_info["height"],
        len(unique_codes), tile_info["size_mb"],
    )
    logger.info("Fire-vegetation classes present: %s",
                tile_info["fire_veg_codes_present"])

    if not tile_info["fire_veg_codes_present"]:
        logger.warning(
            "No fire-vegetation classes (10/20/30) found. "
            "Verify raster alignment before T05."
        )

    # ── Write manifest ────────────────────────────────────────────────────────
    mf = {
        "generated":     datetime.now(timezone.utc).isoformat(),
        "source":        "ESA WorldCover 2021 v200",
        "provider":      "European Space Agency (ESA) / Copernicus",
        "download_path": _WC_S3_BASE,
        "license":       "CC BY 4.0 — ESA WorldCover 2021 v200",
        "vintage":       "2021 reference year",
        "tiles_used":    tile_ids,
        "aoi_bbox_wsen": bbox,
        "substitution_note": (
            "ESA WorldCover 2021 substituted for CORINE CLC 2018. "
            "EEA WMS GetMap returns blank tiles for this AOI; "
            "ImageServer ExportImage endpoint returns 404. "
            "WorldCover provides equivalent fire-vegetation classification "
            "at higher resolution (10 m vs 100 m CORINE). "
            "Class mapping: WC10=forest, WC20=shrubland, WC30=grassland."
        ),
        "tile": tile_info,
    }
    manifest.write_text(json.dumps(mf, indent=2, default=str))

    return out_tif


def main(force: bool = False) -> None:
    """Run the land cover acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_dir = resolve_path("data/raw/corine")

    out_tif = download_corine(output_dir, cfg, force=force)
    print(f"\nOutput: {out_tif}")
    print(f"Size:   {out_tif.stat().st_size / 1024 / 1024:.1f} MB")

    with rasterio.open(out_tif) as ds:
        data = ds.read(1)
        codes = sorted(int(v) for v in np.unique(data) if v > 0)
        print(f"CRS:    {ds.crs}")
        print(f"Size:   {ds.width}x{ds.height} px")
        print(f"Classes present: {codes}")
        print(f"Fire-veg classes: {[c for c in codes if c in FIRE_VEG_CODES]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Download ESA WorldCover 2021 land cover for Attica AOI."
    )
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file exists.")
    args = p.parse_args()
    main(force=args.force)
