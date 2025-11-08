"""
src/acquire/dem.py
------------------
Download Copernicus GLO-30 DEM tiles covering the Attica AOI.

Output
------
data/raw/dem/
  Copernicus_DSM_COG_10_N37_00_E023_00_DEM.tif   (~12 MB)
  Copernicus_DSM_COG_10_N37_00_E024_00_DEM.tif   (~ 5 MB)
  Copernicus_DSM_COG_10_N38_00_E023_00_DEM.tif   (~30 MB)
  Copernicus_DSM_COG_10_N38_00_E024_00_DEM.tif   (~ 7 MB)
  manifest.json                                   (provenance record)

Source
------
Primary: AWS Open Data Registry public S3 bucket (copernicus-dem-30m, eu-central-1).
  - Tiles are Cloud Optimized GeoTIFFs (COGs) in EPSG:4326.
  - No authentication required; bucket is publicly readable.
  - Discovery via Element84 Earth Search STAC v1
    (https://earth-search.aws.element84.com/v1).

Fallback: Copernicus Data Space Ecosystem (CDSE) OData API.
  - Requires valid credentials under 'CopernicusDataSpace' in
    Windows Credential Manager (user/pass → OAuth2 Bearer token).
  - Products are in DTED2 format (not COG); format differs from primary.
  - Activated automatically if primary S3 download fails.

CRS / Resolution / Cadence notes
---------------------------------
- CRS: EPSG:4326 (WGS84 geographic) — tile grid is 1° × 1°
- Pixel size: ~0.0002778°, corresponding to ~30 m at 38°N latitude
  (exact: 1/3600°, i.e. 1 arc-second)
- Version: Copernicus GLO-30 (TanDEM-X based), released 2021
- Temporal: static (no temporal dimension; single acquisition mosaic)
- Vertical datum: EGM2008 (geoid, metres)

Caveats
-------
- GLO-30 is distributed as separate 1°×1° tiles; all 4 tiles must be
  mosaicked before computing slope/aspect at tile boundaries (T05).
- Some tiles over ocean/flat areas are very small (pure 0s); this is
  expected (N37_E024 covers mostly sea east of Athens).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.utils.config import get_bbox, load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAC_URL = "https://earth-search.aws.element84.com/v1/search"
_S3_BASE  = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"

# The 4 tiles required to cover the Attica AOI bbox [23.4, 37.6, 24.2, 38.3]
# Named by lower-left (SW) corner of each 1°×1° cell.
_TILE_IDS = [
    "Copernicus_DSM_COG_10_N37_00_E023_00_DEM",
    "Copernicus_DSM_COG_10_N37_00_E024_00_DEM",
    "Copernicus_DSM_COG_10_N38_00_E023_00_DEM",
    "Copernicus_DSM_COG_10_N38_00_E024_00_DEM",
]

_CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
_CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_CDSE_ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


# ---------------------------------------------------------------------------
# Windows Credential Manager helper (CDSE fallback only)
# ---------------------------------------------------------------------------

def _read_wincred(target: str) -> tuple[str, str]:
    """Read username/password from Windows Credential Manager.

    Returns (username, password).
    Raises RuntimeError if the credential is not found.
    """
    import ctypes
    import ctypes.wintypes

    CRED_TYPE_GENERIC = 1

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime",  ctypes.wintypes.DWORD),
            ("dwHighDateTime", ctypes.wintypes.DWORD),
        ]

    class _CRED_ATTR(ctypes.Structure):
        _fields_ = [
            ("Keyword",   ctypes.wintypes.LPWSTR),
            ("Flags",     ctypes.wintypes.DWORD),
            ("ValueSize", ctypes.wintypes.DWORD),
            ("Value",     ctypes.POINTER(ctypes.c_byte)),
        ]

    class _CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags",              ctypes.wintypes.DWORD),
            ("Type",               ctypes.wintypes.DWORD),
            ("TargetName",         ctypes.wintypes.LPWSTR),
            ("Comment",            ctypes.wintypes.LPWSTR),
            ("LastWritten",        _FILETIME),
            ("CredentialBlobSize", ctypes.wintypes.DWORD),
            ("CredentialBlob",     ctypes.POINTER(ctypes.c_byte)),
            ("Persist",            ctypes.wintypes.DWORD),
            ("AttributeCount",     ctypes.wintypes.DWORD),
            ("Attributes",         ctypes.POINTER(_CRED_ATTR)),
            ("TargetAlias",        ctypes.wintypes.LPWSTR),
            ("UserName",           ctypes.wintypes.LPWSTR),
        ]

    advapi32 = ctypes.windll.advapi32
    pcred = ctypes.POINTER(_CREDENTIAL)()
    if not advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pcred)):
        err = ctypes.get_last_error()
        raise RuntimeError(
            f"Credential '{target}' not found in Windows Credential Manager "
            f"(winerror={err}). "
            "Create it via Control Panel > Credential Manager > "
            "Windows Credentials > Add a generic credential."
        )
    cred = pcred.contents
    username = cred.UserName
    blob = bytes(cred.CredentialBlob[: cred.CredentialBlobSize])
    password = blob.decode("utf-16-le")
    advapi32.CredFree(pcred)
    return username, password


# ---------------------------------------------------------------------------
# Primary download path: AWS Open Data Registry
# ---------------------------------------------------------------------------

def _stac_discover_tiles(bbox: list[float]) -> dict[str, str]:
    """Query Earth Search STAC to discover COG tile HTTPS URLs.

    Args:
        bbox: [west, south, east, north] in WGS84.

    Returns:
        Dict mapping tile_id → HTTPS URL (direct download link).
    """
    logger.info("Querying Earth Search STAC for cop-dem-glo-30 tiles...")
    resp = requests.post(
        _STAC_URL,
        json={
            "collections": ["cop-dem-glo-30"],
            "bbox": bbox,
            "limit": 20,
        },
        timeout=30,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    logger.info("STAC returned %d tile candidates.", len(features))

    tile_urls: dict[str, str] = {}
    for feat in features:
        fid = feat.get("id", "")
        # Asset 'data' holds the COG href (s3:// URL)
        href = feat.get("assets", {}).get("data", {}).get("href", "")
        if href.startswith("s3://"):
            # Convert s3://bucket/key to HTTPS using eu-central-1 path
            parts = href.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            https_url = f"https://{bucket}.s3.eu-central-1.amazonaws.com/{key}"
        elif href.startswith("http"):
            https_url = href
        else:
            continue
        tile_urls[fid] = https_url

    return tile_urls


def _download_tile_s3(
    tile_id: str,
    url: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Download a single GLO-30 COG tile from S3 with progress logging.

    Args:
        tile_id: Tile name (used as output filename stem).
        url:     HTTPS URL to the GeoTIFF.
        output_dir: Directory to save the tile.
        force:   Re-download even if file already exists.

    Returns:
        Path to the downloaded file.
    """
    out_path = output_dir / f"{tile_id}.tif"
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("Already exists: %s (%.1f MB)", out_path.name, size_mb)
        return out_path

    logger.info("Downloading %s ...", tile_id)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    downloaded = 0
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info("Saved: %s (%.1f MB)", out_path.name, size_mb)
    return out_path


# ---------------------------------------------------------------------------
# Fallback download path: CDSE OData + zipper
# ---------------------------------------------------------------------------

def _get_cdse_token() -> str:
    """Get CDSE OAuth2 Bearer token from Windows Credential Manager."""
    username, password = _read_wincred("CopernicusDataSpace")
    resp = requests.post(
        _CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("CDSE OAuth2 token response missing access_token")
    logger.info("CDSE OAuth2 token acquired (expires_in=%ds).",
                resp.json().get("expires_in", 0))
    return token


def _cdse_search_products(bbox: list[float], token: str) -> list[dict[str, Any]]:
    """Search CDSE OData for COP-DEM products intersecting the AOI bbox."""
    west, south, east, north = bbox
    polygon = (
        f"POLYGON(({west} {south},{east} {south},{east} {north},"
        f"{west} {north},{west} {south}))"
    )
    resp = requests.get(
        _CDSE_ODATA_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "$filter": (
                "Collection/Name eq 'COP-DEM' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
            ),
            "$top": "30",
            "$select": "Id,Name,S3Path,ContentLength",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


def _cdse_download_product(
    product_id: str,
    product_name: str,
    output_dir: Path,
    token: str,
    force: bool = False,
) -> Path:
    """Download a CDSE product ZIP and extract to output_dir."""
    out_zip = output_dir / f"{product_name}.zip"
    if out_zip.exists() and not force:
        logger.info("CDSE product ZIP already exists: %s", out_zip.name)
        return out_zip

    url = f"{_CDSE_ZIPPER_URL}({product_id})/$value"
    logger.info("Downloading CDSE product: %s ...", product_name)
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    with open(out_zip, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)

    size_mb = out_zip.stat().st_size / 1024 / 1024
    logger.info("Saved: %s (%.1f MB)", out_zip.name, size_mb)
    return out_zip


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_tiles(output_dir: Path, bbox: list[float]) -> dict[str, Any]:
    """Open each downloaded tile and verify it covers the expected area.

    Returns a dict with per-tile stats.
    """
    import rasterio

    west, south, east, north = bbox
    stats: dict[str, Any] = {}

    for tif in sorted(output_dir.glob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                b = ds.bounds
                tile_stats = {
                    "crs": str(ds.crs),
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": [b.left, b.bottom, b.right, b.top],
                    "dtype": str(ds.dtypes[0]),
                    "nodata": ds.nodata,
                    "size_mb": round(tif.stat().st_size / 1024 / 1024, 2),
                }
                stats[tif.name] = tile_stats
                logger.info(
                    "  %s: %dx%d px, bounds=[%.2f,%.2f,%.2f,%.2f], %.1f MB",
                    tif.name, ds.width, ds.height,
                    b.left, b.bottom, b.right, b.top,
                    tile_stats["size_mb"],
                )
        except Exception as exc:
            logger.warning("Could not open %s: %s", tif.name, exc)
            stats[tif.name] = {"error": str(exc)}

    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_dem(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> list[Path]:
    """Download Copernicus GLO-30 DEM tiles covering the Attica AOI.

    Attempts primary (AWS S3 public COG) first; falls back to CDSE if
    S3 tiles are unavailable for any reason.

    Args:
        output_dir: Target directory (created if missing).
        cfg:        Merged pipeline config (from load_config()).
        force:      Re-download even if files already exist.

    Returns:
        List of downloaded .tif file paths.

    Raises:
        RuntimeError: If all download paths fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = get_bbox(cfg)  # [west, south, east, north]

    # ── Primary: Earth Search STAC + AWS S3 ──────────────────────────────────
    logger.info("--- DEM primary: Earth Search STAC + AWS S3 ---")
    downloaded_paths: list[Path] = []
    s3_failed_tiles: list[str] = []

    try:
        tile_urls = _stac_discover_tiles(bbox)
    except Exception as exc:
        logger.warning("STAC discovery failed: %s — will try known tile IDs.", exc)
        # Construct URLs from known tile IDs without STAC
        tile_urls = {
            tid: f"{_S3_BASE}/{tid}/{tid}.tif"
            for tid in _TILE_IDS
        }

    for tile_id in _TILE_IDS:
        url = tile_urls.get(tile_id)
        if not url:
            # Try constructing URL directly from tile_id
            url = f"{_S3_BASE}/{tile_id}/{tile_id}.tif"
            logger.info("Tile %s not in STAC results — using constructed URL.", tile_id)

        try:
            path = _download_tile_s3(tile_id, url, output_dir, force=force)
            downloaded_paths.append(path)
        except Exception as exc:
            logger.warning("S3 download failed for %s: %s", tile_id, exc)
            s3_failed_tiles.append(tile_id)

    # ── Fallback: CDSE OData + zipper ────────────────────────────────────────
    if s3_failed_tiles:
        logger.warning(
            "%d tile(s) failed via S3: %s — trying CDSE fallback.",
            len(s3_failed_tiles), s3_failed_tiles,
        )
        try:
            token = _get_cdse_token()
            products = _cdse_search_products(bbox, token)
            logger.info("CDSE found %d COP-DEM products.", len(products))
            for product in products:
                pid = product.get("Id")
                pname = product.get("Name", f"cdse_{pid}")
                try:
                    p = _cdse_download_product(pid, pname, output_dir, token, force=force)
                    if p not in downloaded_paths:
                        downloaded_paths.append(p)
                except Exception as exc:
                    logger.warning("CDSE download failed for %s: %s", pname, exc)
        except Exception as exc:
            logger.error("CDSE fallback failed: %s", exc)

    if not downloaded_paths:
        raise RuntimeError(
            "DEM download failed — no tiles retrieved via S3 or CDSE. "
            "Check connectivity and Windows Credential Manager 'CopernicusDataSpace'."
        )

    # ── Verify coverage ───────────────────────────────────────────────────────
    tif_paths = [p for p in downloaded_paths if p.suffix == ".tif"]
    logger.info("Verifying %d downloaded tiles...", len(tif_paths))
    stats = _verify_tiles(output_dir, bbox)

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "Copernicus GLO-30 DEM, AWS Open Data Registry (copernicus-dem-30m)",
        "license": "Copernicus DEM open license (CC-BY-4.0)",
        "resolution_arcsec": 1,
        "resolution_m_approx": 30,
        "crs": "EPSG:4326",
        "tile_naming": "Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM",
        "tiles": stats,
        "aoi_bbox_wsen": bbox,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written: %s", manifest_path)

    logger.info("DEM download complete: %d file(s) in %s", len(tif_paths), output_dir)
    return tif_paths


def main(force: bool = False) -> None:
    """Run the DEM acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_dir = resolve_path(cfg["sources"]["dem"]["output_dir"])

    paths = download_dem(output_dir, cfg, force=force)

    print(f"\nDownloaded {len(paths)} DEM file(s):")
    for p in sorted(paths):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:55s} {size_mb:6.1f} MB")
    print(f"\nOutput dir: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Copernicus GLO-30 DEM tiles for Attica AOI."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist.")
    args = parser.parse_args()
    main(force=args.force)
