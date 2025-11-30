"""
preprocess/vegetation.py — Reclassify WorldCover to binary fire-vegetation mask.

Fire-relevant ESA WorldCover classes (uint8):
  10 = Tree cover      → 1
  20 = Shrubland       → 1
  30 = Grassland       → 1
  all others           → 0

Inputs:
  data/raw/corine/landcover_attica_2021.tif   (WorldCover, uint8, EPSG:4326, 10 m)

Outputs:
  data/processed/corine_binary_epsg2100.tif   (uint8, {0,1}, EPSG:2100, 10 m)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import resolve_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIRE_VEG_CODES = {10, 20, 30}   # WorldCover codes → fire vegetation = 1
TARGET_RES_M   = 10.0            # preserve native 10 m resolution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = resolve_path(".")

    in_path  = root / "data/raw/corine/landcover_attica_2021.tif"
    out_path = root / "data/processed/corine_binary_epsg2100.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crs_work = CRS.from_epsg(2100)

    # ------------------------------------------------------------------
    # 1. Read source raster
    # ------------------------------------------------------------------
    print(f"[vegetation] Reading {in_path.name} ...")
    with rasterio.open(in_path) as src:
        src_crs       = src.crs
        src_transform = src.transform
        src_nodata    = src.nodata
        raw = src.read(1)

    print(f"  shape : {raw.shape}  CRS : {src_crs}")
    print(f"  unique values : {np.unique(raw).tolist()}")

    # ------------------------------------------------------------------
    # 2. Reclassify to binary fire-vegetation mask (in source CRS)
    # ------------------------------------------------------------------
    print(f"[vegetation] Reclassifying codes {sorted(FIRE_VEG_CODES)} → 1, rest → 0 ...")
    binary = np.zeros(raw.shape, dtype="uint8")
    for code in FIRE_VEG_CODES:
        binary[raw == code] = 1

    # Where original nodata (0) exists, set binary output nodata = 255
    # (nodata=0 in source means "no WorldCover data", not "non-vegetated")
    if src_nodata is not None:
        nodata_mask = raw == int(src_nodata)
        binary[nodata_mask] = 255

    veg_frac = (binary == 1).sum() / (binary != 255).sum()
    print(f"  fire-veg fraction : {veg_frac:.3f}  (1={FIRE_VEG_CODES}, 0=other, 255=nodata)")

    # ------------------------------------------------------------------
    # 3. Reproject to EPSG:2100 at 10 m (nearest-neighbour — categorical)
    # ------------------------------------------------------------------
    print(f"[vegetation] Reprojecting to EPSG:2100 at {TARGET_RES_M} m ...")

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        crs_work,
        raw.shape[1],
        raw.shape[0],
        left=src_transform.c,
        bottom=src_transform.f + src_transform.e * raw.shape[0],
        right=src_transform.c + src_transform.a * raw.shape[1],
        top=src_transform.f,
        resolution=TARGET_RES_M,
    )

    binary_2100 = np.full((dst_height, dst_width), 255, dtype="uint8")

    reproject(
        source=binary,
        destination=binary_2100,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=255,
        dst_transform=dst_transform,
        dst_crs=crs_work,
        dst_nodata=255,
        resampling=Resampling.nearest,
    )

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    profile = {
        "driver": "GTiff",
        "crs": crs_work,
        "transform": dst_transform,
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary_2100[np.newaxis, ...])

    # ------------------------------------------------------------------
    # 5. QC
    # ------------------------------------------------------------------
    valid_mask = binary_2100 != 255
    unique_valid = np.unique(binary_2100[valid_mask])
    frac_fire = (binary_2100[valid_mask] == 1).sum() / valid_mask.sum()

    print("\n[vegetation] QC summary:")
    print(f"  output shape   : {binary_2100.shape}")
    print(f"  unique values  : {np.unique(binary_2100).tolist()}  (valid: {unique_valid.tolist()})")
    print(f"  valid pixels   : {valid_mask.sum():,} / {valid_mask.size:,}")
    print(f"  fire-veg frac  : {frac_fire:.3f}")

    assert set(unique_valid).issubset({0, 1}), \
        f"Unexpected unique values in binary output: {unique_valid}"
    print("  [check] unique valid values ⊆ {0, 1} — OK")
    print(f"\n[vegetation] Done → {out_path.name}")


if __name__ == "__main__":
    main()
