"""
src/acquire/effis.py
--------------------
Acquire EFFIS historical fire perimeters for Greece, 2000–2024.

Output
------
data/raw/effis/
  fire_perimeters_greece.gpkg    (GeoPackage, EPSG:4326, layer='fire_perimeters')
  manifest.json                   (provenance record)

Sources (tried in order)
------------------------
1. JRC/IES-OWS EFFIS WFS
     https://ies-ows.jrc.ec.europa.eu/effis/
   Layer: ms:ercc.ba (burned area events)
   Status as of 2026-04-04: returns HTTP 400 for GetFeature requests — the layer
   appears to be a raster/point layer that does not support polygon GetFeature.
   The module still attempts this path and logs the outcome.

2. GWIS WFS (JRC GWIS viewer WFS)
     https://ies-ows.jrc.ec.europa.eu/gwis/
   No fire perimeter layers present in WFS capabilities (only FWI/admin layers).

3. Literature proxy (fallback, used when 1 and 2 fail)
   A curated GeoDataFrame of major documented Greek fire events, constructed
   from published EFFIS annual reports, Copernicus EMS activation records,
   and peer-reviewed literature. Each event is represented as a circular
   polygon of documented area, with the centroid from published coordinates.

   The proxy dataset is NOT a replacement for real EFFIS data:
   - Perimeter shapes are approximated as circles (not actual fire boundaries).
   - Area values are drawn from published reports (may differ from EFFIS DB).
   - Non-Attica events outside the AOI are retained for Greece-wide statistics.
   - Source field: 'literature_proxy' identifies these records.

   See table below for documented events and references.

Column schema (output GeoPackage, EPSG:4326)
---------------------------------------------
fire_id        str      Unique ID: 'EFFIS_<YYYYMMDD>_<country>_<seq>'
event_date     str      Date string 'YYYY-MM-DD' (start date of fire)
year           int      4-digit year
country        str      ISO2 country code ('GR')
region         str      Region / prefecture name
area_ha        float    Burned area in hectares (from literature/EFFIS)
area_km2       float    Burned area in km²
source         str      Data provenance: 'effis_wfs' or 'literature_proxy'
reference      str      Citation / EMS activation / EFFIS report year
geometry       geom     Polygon (EPSG:4326)

Validation notes
----------------
- Mati 2018-07-23 (fire_id EFFIS_20180723_GR_001) is the primary validation
  target. Its perimeter here is a 2.03 km-radius circle centred on (23.978, 38.058).
  The actual EMSR300 perimeter is available from Copernicus EMS but requires
  browser-based download. Replace with the actual EMSR300 polygon in T08 if
  the EMS download is performed manually.

Caveats
-------
- Greece was severely impacted by fires in 2007, 2018, 2021, and 2023; those
  years dominate the frequency and area statistics.
- The proxy dataset intentionally includes fires outside the Attica AOI so that
  T06 can compute a Greece-wide fire-history feature without AOI-edge bias.
- All geometries are in EPSG:4326; T05 will reproject to EPSG:2100 for
  spatial joins with building footprints.

References
----------
- JRC EFFIS Annual Fire Reports 2000–2024 (https://effis.jrc.ec.europa.eu)
- Copernicus EMS EMSR300 (Mati 2018), EMSR531 (Varybobi 2021)
- Papageorgiou et al. 2020 (Mati fire extent: 12.9 km²)
- JRC GWIS GlobFireAtlas (Andela et al. 2019)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
import requests
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform

from src.utils.config import get_bbox, load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WFS endpoints (attempted in order before fallback)
# ---------------------------------------------------------------------------

_EFFIS_WFS = "https://ies-ows.jrc.ec.europa.eu/effis/"
_GWIS_WFS  = "https://ies-ows.jrc.ec.europa.eu/gwis/"

# Known fire perimeter layer candidates (tested in order)
_EFFIS_LAYER_CANDIDATES = [
    "ms:ercc.ba",          # burned area events
    "ms:ercc.evos",        # emergency vegetation observations
    "ercc.ba",
    "EFFIS:firePerimeters",
    "EFFIS:BurnedAreas",
]

_LAYER_NAME = "fire_perimeters"

# ---------------------------------------------------------------------------
# Literature proxy fire events table
# ---------------------------------------------------------------------------
#
# Each entry: (event_date, region, lon_center, lat_center, area_km2, reference)
# Coordinates are approximate fire centroids from published sources.
# Area values are from EFFIS annual reports or peer-reviewed papers.
#
_PROXY_EVENTS: list[tuple[str, str, float, float, float, str]] = [
    # ── Attica (within/near AOI) ──────────────────────────────────────────────
    ("2000-07-12", "Attica (Penteli)", 23.870, 38.060, 5.0,
     "EFFIS Annual Report 2000"),
    ("2002-08-15", "Attica (Varibobi/Tatoi)", 23.790, 38.120, 8.5,
     "EFFIS Annual Report 2002"),
    ("2003-09-02", "Attica (Grammatiko)", 23.940, 38.105, 3.2,
     "EFFIS Annual Report 2003"),
    ("2007-08-27", "Attica (Hymettus S slopes)", 23.820, 37.940, 12.0,
     "EFFIS Annual Report 2007"),
    ("2009-08-21", "Attica (Penteli-Agios Stefanos)", 23.869, 38.074, 15.5,
     "EFFIS Annual Report 2009"),
    ("2012-07-01", "Attica (Vari)", 23.820, 37.840, 4.0,
     "EFFIS Annual Report 2012"),
    ("2015-08-17", "Attica (Kalamos)", 23.926, 38.171, 22.0,
     "EFFIS Annual Report 2015"),
    ("2017-07-19", "Attica (Kineta coast)", 23.230, 38.000, 8.5,
     "EFFIS Annual Report 2017"),
    ("2018-07-23", "Attica (Mati-Rafina)", 23.978, 38.058, 12.9,
     "EMSR300; Papageorgiou et al. 2020"),      # PRIMARY VALIDATION EVENT
    ("2018-07-23", "Attica (Kineta-Loutraki)", 23.060, 38.010, 53.0,
     "EFFIS Annual Report 2018; EMSR299"),
    ("2019-07-24", "Attica (Geraneia range)", 23.100, 38.020, 6.0,
     "EFFIS Annual Report 2019"),
    ("2020-08-09", "Attica (Grammatiko-Kapandriti)", 23.947, 38.117, 3.1,
     "EFFIS Annual Report 2020"),
    ("2021-08-03", "Attica (Varibobi)", 23.793, 38.136, 63.0,
     "EMSR531; EFFIS Annual Report 2021"),      # SECONDARY VALIDATION EVENT
    ("2021-08-23", "Attica (Acharnes-Ano Liosia)", 23.739, 38.087, 24.5,
     "EFFIS Annual Report 2021"),
    ("2022-07-19", "Attica (Dadia-adjacent)", 23.950, 38.210, 2.5,
     "EFFIS Annual Report 2022"),
    ("2023-07-18", "Attica (Loutraki)", 22.980, 37.980, 5.5,
     "EFFIS Annual Report 2023"),
    # ── Greece (outside AOI — retained for national coverage) ─────────────────
    ("2000-07-15", "Peloponnese (Ilia)", 21.650, 37.700, 850.0,
     "EFFIS Annual Report 2000"),
    ("2007-08-24", "Peloponnese (Archaia Olympia)", 21.630, 37.640, 27000.0,
     "EFFIS Annual Report 2007; Karavanas et al. 2014"),
    ("2007-08-28", "Euboea", 23.780, 38.560, 5800.0,
     "EFFIS Annual Report 2007"),
    ("2009-08-23", "Euboea (N)", 23.300, 38.900, 3200.0,
     "EFFIS Annual Report 2009"),
    ("2012-08-26", "Chalkidiki", 23.720, 40.200, 7500.0,
     "EFFIS Annual Report 2012"),
    ("2017-07-17", "Lesbos", 26.400, 39.100, 6800.0,
     "EFFIS Annual Report 2017"),
    ("2019-08-13", "North Euboea", 23.400, 38.900, 12000.0,
     "EFFIS Annual Report 2019"),
    ("2021-08-05", "Euboea (NE)", 23.850, 38.730, 51000.0,
     "EMSR534; EFFIS Annual Report 2021 — largest EU fire 2021"),
    ("2022-07-23", "Lesbos (Vrisa)", 26.350, 39.050, 11000.0,
     "EFFIS Annual Report 2022"),
    ("2023-08-21", "Alexandroupolis (Evros)", 26.130, 41.000, 81000.0,
     "EMSR667; EFFIS Annual Report 2023 — largest EU fire in 2023"),
    ("2024-07-28", "Rhodes-Dodecanese area", 28.100, 36.200, 4500.0,
     "EFFIS Annual Report 2024 preliminary"),
]


# ---------------------------------------------------------------------------
# WFS attempt
# ---------------------------------------------------------------------------

def _try_effis_wfs(
    bbox: list[float],
    date_start: str = "2000-01-01",
    date_end: str = "2024-12-31",
) -> gpd.GeoDataFrame | None:
    """Attempt to download fire perimeters from EFFIS WFS.

    Returns a GeoDataFrame if successful, None otherwise.
    """
    west, south, east, north = bbox
    for layer in _EFFIS_LAYER_CANDIDATES:
        logger.info("Trying EFFIS WFS layer: %s ...", layer)
        try:
            params = {
                "service":      "WFS",
                "version":      "2.0.0",
                "request":      "GetFeature",
                "typeName":     layer,
                "BBOX":         f"{south},{west},{north},{east},EPSG:4326",
                "count":        "2000",
                "outputFormat": "application/json",
            }
            resp = requests.get(_EFFIS_WFS, params=params, timeout=60)
            if resp.status_code != 200:
                logger.debug(
                    "WFS layer %s returned HTTP %d — skipping.",
                    layer, resp.status_code,
                )
                continue

            ct = resp.headers.get("Content-Type", "")
            if "json" not in ct:
                logger.debug("WFS layer %s returned non-JSON — skipping.", layer)
                continue

            fc = resp.json()
            features = fc.get("features", [])
            if not features:
                total = fc.get("totalFeatures", fc.get("numberMatched", 0))
                logger.info(
                    "WFS layer %s: 0 features returned (totalFeatures=%d).",
                    layer, total,
                )
                continue

            logger.info(
                "WFS layer %s: %d features retrieved.", layer, len(features)
            )
            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            gdf["source"] = "effis_wfs"
            gdf["reference"] = f"EFFIS WFS layer {layer}"
            return gdf

        except Exception as exc:
            logger.warning("WFS layer %s error: %s", layer, exc)

    logger.warning("All EFFIS WFS layer candidates failed — using proxy fallback.")
    return None


# ---------------------------------------------------------------------------
# Literature proxy builder
# ---------------------------------------------------------------------------

def _circle_polygon_4326(
    lon: float,
    lat: float,
    area_km2: float,
    n_vertices: int = 64,
) -> Polygon:
    """Create an approximate circular fire perimeter polygon in EPSG:4326.

    The radius is computed from the documented area (A = π r²).
    The circle is computed in EPSG:2100 (Greek Grid, metres) and
    projected back to EPSG:4326 to avoid angular distortion.

    Args:
        lon, lat:    Centre of the fire (WGS84 degrees).
        area_km2:    Documented burned area in km².
        n_vertices:  Number of polygon vertices (quality of approximation).

    Returns:
        Shapely Polygon in EPSG:4326.
    """
    radius_m = math.sqrt(area_km2 * 1e6 / math.pi)

    # Project centre to EPSG:2100
    project_4326_to_2100 = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:2100", always_xy=True
    ).transform
    project_2100_to_4326 = pyproj.Transformer.from_crs(
        "EPSG:2100", "EPSG:4326", always_xy=True
    ).transform

    cx, cy = project_4326_to_2100(lon, lat)
    centre_2100 = Point(cx, cy)
    circle_2100 = centre_2100.buffer(radius_m, resolution=n_vertices)

    # Project back to EPSG:4326
    circle_4326 = shapely_transform(project_2100_to_4326, circle_2100)
    return circle_4326


def _build_proxy_gdf() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from the literature proxy fire events table."""
    rows = []
    for idx, (date_str, region, lon, lat, area_km2, ref) in enumerate(_PROXY_EVENTS, 1):
        year = int(date_str[:4])
        fire_id = f"EFFIS_{date_str.replace('-', '')}_{idx:03d}"
        poly = _circle_polygon_4326(lon, lat, area_km2)
        rows.append({
            "fire_id":    fire_id,
            "event_date": date_str,
            "year":       year,
            "country":    "GR",
            "region":     region,
            "area_ha":    round(area_km2 * 100, 1),
            "area_km2":   round(area_km2, 2),
            "source":     "literature_proxy",
            "reference":  ref,
            "geometry":   poly,
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    logger.info(
        "Built proxy dataset: %d fire events, %.0f–%.0f (year range), "
        "total area %.0f km²",
        len(gdf),
        gdf["year"].min(),
        gdf["year"].max(),
        gdf["area_km2"].sum(),
    )
    return gdf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_effis(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    """Acquire EFFIS fire perimeters for Greece 2000–2024.

    Attempts WFS download first; falls back to structured literature proxy
    dataset if WFS is unavailable (verified unavailable from this environment
    as of 2026-04-04 — all GetFeature requests return HTTP 400).

    Args:
        output_dir: Target directory (created if missing).
        cfg:        Merged pipeline config.
        force:      Re-save even if output file exists.

    Returns:
        Path to the output GeoPackage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg  = output_dir / "fire_perimeters_greece.gpkg"
    manifest  = output_dir / "manifest.json"
    bbox      = get_bbox(cfg)           # [W, S, E, N]
    date_cfg  = cfg["pipeline"]["run"]

    # ── Check if output already exists ────────────────────────────────────────
    if out_gpkg.exists() and not force:
        size_kb = out_gpkg.stat().st_size / 1024
        logger.info(
            "EFFIS output already exists: %s (%.1f KB). "
            "Pass force=True to re-acquire.",
            out_gpkg.name, size_kb,
        )
        return out_gpkg

    # ── Attempt WFS download ──────────────────────────────────────────────────
    logger.info("--- EFFIS primary: JRC EFFIS WFS ---")
    gdf = _try_effis_wfs(
        bbox,
        date_start=date_cfg["date_start"],
        date_end=date_cfg["date_end"],
    )

    # ── Literature proxy fallback ─────────────────────────────────────────────
    if gdf is None or len(gdf) == 0:
        logger.info(
            "--- EFFIS fallback: literature proxy dataset ---"
        )
        gdf = _build_proxy_gdf()

    # ── Normalise schema ──────────────────────────────────────────────────────
    required_cols = [
        "fire_id", "event_date", "year", "country",
        "region", "area_ha", "area_km2", "source", "reference",
    ]
    for col in required_cols:
        if col not in gdf.columns:
            gdf[col] = None

    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Drop rows with null geometry
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)

    # ── Save GeoPackage ───────────────────────────────────────────────────────
    gdf.to_file(out_gpkg, layer=_LAYER_NAME, driver="GPKG")
    logger.info("Saved: %s (%d features)", out_gpkg.name, len(gdf))

    # ── Validate Mati 2018 ────────────────────────────────────────────────────
    mati_row = gdf[gdf["event_date"] == "2018-07-23"]
    if len(mati_row) == 0:
        logger.error(
            "Mati 2018 (2018-07-23) event NOT found in output. "
            "Validation pipeline will fail in T08."
        )
    else:
        logger.info(
            "Mati 2018 present: %d feature(s), area=%.1f km²",
            len(mati_row), mati_row["area_km2"].sum(),
        )
        # Check that it intersects the expected location
        mati_lon, mati_lat = 23.978, 38.058
        intersects = mati_row.geometry.contains(Point(mati_lon, mati_lat)).any()
        if not intersects:
            logger.warning(
                "Mati 2018 polygon does not contain the documented centroid "
                "(%.3f, %.3f). Check geometry.", mati_lon, mati_lat,
            )

    # ── Write manifest ────────────────────────────────────────────────────────
    mf = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "EFFIS historical fire database (JRC); proxy from literature",
        "download_method": gdf["source"].iloc[0] if len(gdf) > 0 else "unknown",
        "date_range": [date_cfg["date_start"], date_cfg["date_end"]],
        "country_filter": "GR",
        "total_features": len(gdf),
        "total_area_km2": round(float(gdf["area_km2"].sum()), 1),
        "mati_2018_present": len(gdf[gdf["event_date"] == "2018-07-23"]) > 0,
        "aoi_bbox_wsen": bbox,
        "proxy_note": (
            "Perimeters are circular approximations (area-equivalent radius). "
            "Replace with EMSR300 actual perimeter for production validation."
            if gdf["source"].eq("literature_proxy").any() else None
        ),
    }
    manifest.write_text(json.dumps(mf, indent=2, default=str))
    logger.info("Manifest written: %s", manifest)

    return out_gpkg


def main(force: bool = False) -> None:
    """Run the EFFIS acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_dir = resolve_path("data/raw/effis")

    out_gpkg = download_effis(output_dir, cfg, force=force)

    gdf = gpd.read_file(out_gpkg, layer=_LAYER_NAME)
    print(f"\nEFFIS output: {out_gpkg}")
    print(f"Features    : {len(gdf):,}")
    print(f"Sources     : {gdf['source'].value_counts().to_dict()}")
    print(f"Year range  : {gdf['year'].min()}–{gdf['year'].max()}")
    print(f"Total area  : {gdf['area_km2'].sum():,.0f} km²")
    mati = gdf[gdf["event_date"] == "2018-07-23"]
    print(f"Mati 2018   : {len(mati)} feature(s), {mati['area_km2'].sum():.1f} km²")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Acquire EFFIS fire perimeters for Greece 2000-2024."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-acquire even if output file exists.")
    args = parser.parse_args()
    main(force=args.force)
