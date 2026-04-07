"""
src/acquire/buildings.py
------------------------
Download OSM building footprints for the Attica WUI and validation coverage
area via Overpass API.

Output
------
data/raw/buildings_osm_attica.gpkg  (GeoPackage, layer='buildings', EPSG:4326)

Download scope
--------------
The full Attica AOI bbox contains ~413,000 buildings (verified 2026-04-04).
Dense urban central Athens (~330k buildings) is excluded here because those
buildings are non-WUI and would dilute the analysis without analytical value.
The download covers five geographic chunks (defined in config/data_sources.yaml
→ buildings.primary.download_chunks):

  east_attica_ne  [23.8, 37.95, 24.2, 38.3]  ~74,700 bldgs
    Mati (2018 validation area), Rafina, Gerakas, Pallini,
    eastern Hymettus slopes, Agios Stefanos.

  east_attica_se  [23.8, 37.6, 24.2, 37.95]  ~7,800 bldgs
    Markopulo, Porto Rafti, Lavrio — low-density eastern coastal zone.

  north_attica_wui  [23.7, 38.1, 24.0, 38.3]  ~9,500 bldgs
    Varybobi, Tatoi, Agios Stefanos — Parnitha/Penteli forest interface.
    Also contains the Phase 2 Varybobi 2021 validation area.

  central_attica_wui  [23.7, 38.0, 23.8, 38.1]  count TBD
    Acharnes, Ano Liosia, Fyli foothills — Acharnes/Varybobi validation
    overlap zone. Required for Acharnes 2021 validation coverage.

  west_attica_wui  [23.55, 38.0, 23.7, 38.3]  count TBD
    Fyli, Erythres — Varybobi bbox western coverage. Narrowed from
    [23.4, 38.0] to exclude Vilia/Dervenochoria (no validation events).

Excluded
--------
- Central Athens and inner suburbs ([23.7, 37.6, 23.8, 38.0] — Piraeus,
  Peristeri, Ilion, Egaleo, etc.) are not downloaded. They are clearly non-WUI
  and represent ~80% of the total building stock in the AOI.
- West Attica south ([23.4, 37.6, 23.7, 38.0] — Megara, Kineta) is excluded:
  ~36K buildings with no validation event coverage and all scored low-risk.
- Far west Attica north ([23.4, 38.0, 23.55, 38.0] — Vilia, Dervenochoria)
  is excluded: outside all validation event bboxes.

Column schema (GeoPackage, EPSG:4326)
--------------------------------------
building_id   str      'way/<osm_id>' — unique identifier
osm_id        int64    raw OSM numeric ID
osm_type      str      'way' (OSM relations excluded, see note below)
source        str      'osm'
building_tag  str      OSM building= value (e.g. 'yes', 'house', 'apartments')
building_use  str      derived: residential|commercial|civic|industrial|
                       agricultural|utility|unknown
name          str      OSM name= tag, or None
chunk         str      download chunk name (for audit/provenance)
geometry      geom     Polygon, EPSG:4326

Notes / Caveats
---------------
- OSM **way** elements only; multipolygon **relations** are skipped.
  Relations represent <1% of buildings in Greece and require significantly
  more complex parsing (outer/inner ring assembly). Add in a later phase if
  building completeness for large civic/commercial structures is needed.
- Overpass API rate limits may cause transient 504 errors. The downloader
  retries with exponential back-off across two public API instances.
- Building geometry is taken directly from OSM node coordinates — no
  simplification or snapping is applied at this stage. Geometry repairs
  (invalid polygons, small slivers) are handled in T05 (preprocess/buildings.py).
- OSM coverage in peri-urban areas is generally good for Attica but may
  undercount informal/rural structures. Microsoft Building Footprints fallback
  is available if total count < config.run.min_buildings (default: 5,000).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from shapely.geometry import Polygon
from shapely.validation import make_valid

from src.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Overpass API instances (tried in order on failure)
# ---------------------------------------------------------------------------
_OVERPASS_INSTANCES = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

_LAYER_NAME = "buildings"

# ---------------------------------------------------------------------------
# OSM building= tag → simplified use category
# ---------------------------------------------------------------------------
_BUILDING_USE: dict[str, str] = {
    # residential
    "house": "residential",
    "detached": "residential",
    "semidetached_house": "residential",
    "terrace": "residential",
    "apartments": "residential",
    "residential": "residential",
    "bungalow": "residential",
    "dormitory": "residential",
    "flats": "residential",
    "villa": "residential",
    # commercial
    "commercial": "commercial",
    "retail": "commercial",
    "office": "commercial",
    "supermarket": "commercial",
    "hotel": "commercial",
    "shop": "commercial",
    "kiosk": "commercial",
    # civic / public
    "public": "civic",
    "civic": "civic",
    "government": "civic",
    "hospital": "civic",
    "school": "civic",
    "university": "civic",
    "church": "civic",
    "chapel": "civic",
    "mosque": "civic",
    "monastery": "civic",
    "cathedral": "civic",
    "fire_station": "civic",
    "police": "civic",
    "library": "civic",
    "sports_hall": "civic",
    "stadium": "civic",
    # industrial
    "industrial": "industrial",
    "warehouse": "industrial",
    "factory": "industrial",
    "storage_tank": "industrial",
    "storage": "industrial",
    # agricultural
    "barn": "agricultural",
    "farm": "agricultural",
    "farm_auxiliary": "agricultural",
    "greenhouse": "agricultural",
    "silo": "agricultural",
    "stable": "agricultural",
    # utility / infrastructure
    "utility": "utility",
    "transformer_tower": "utility",
    "water_tower": "utility",
    "substation": "utility",
    # generic / unclassified
    "yes": "unknown",
    "no": "unknown",  # should not appear — query filters building=no
}


def _classify_use(building_tag: str, tags: dict[str, str]) -> str:
    """Map OSM building= tag and ancillary tags to a simplified use category.

    Falls back to amenity= tag if building= is generic ('yes').
    """
    use = _BUILDING_USE.get(building_tag.lower())
    if use and use != "unknown":
        return use
    # Check amenity tag for civic buildings tagged building=yes
    amenity = tags.get("amenity", "")
    if amenity in {
        "hospital", "clinic", "school", "university", "college",
        "fire_station", "police", "library", "place_of_worship",
    }:
        return "civic"
    return "unknown"


# ---------------------------------------------------------------------------
# Overpass query helpers
# ---------------------------------------------------------------------------

def _build_query(bbox_wsen: list[float], timeout: int = 300) -> str:
    """Build Overpass QL for building ways in a single bbox.

    Overpass bbox argument order: south, west, north, east.
    We use `out body geom qt;` to include all tags + geometry with
    quicktrack spatial sorting (faster server-side processing).

    Relations (multipolygon buildings) are intentionally excluded — see
    module docstring for rationale.
    """
    west, south, east, north = bbox_wsen
    return (
        f'[out:json][timeout:{timeout}][bbox:{south},{west},{north},{east}];\n'
        'way["building"];\n'
        'out body geom qt;'
    )


def _fetch_overpass(
    query: str,
    max_retries: int = 4,
    base_delay: int = 15,
) -> dict[str, Any]:
    """POST query to Overpass API with retry and instance rotation.

    Strategy:
    - Attempt each API instance in turn before sleeping and retrying.
    - Exponential back-off between retry rounds (15s, 30s, 60s, 120s).
    - A 504 Gateway Timeout indicates server overload; retry after delay.
    - A 429 Too Many Requests means rate-limited; retry after longer delay.
    """
    for attempt in range(1, max_retries + 1):
        for url in _OVERPASS_INSTANCES:
            logger.info("Overpass attempt %d/%d via %s", attempt, max_retries, url)
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    timeout=360,
                    headers={"Accept-Encoding": "gzip"},
                )
                if resp.status_code == 429:
                    wait = base_delay * 4 * attempt
                    logger.warning("Rate-limited (429). Waiting %ds.", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                if not resp.content.strip():
                    logger.warning("Empty response from %s (attempt %d)", url, attempt)
                    continue
                data = resp.json()
                n_elements = len(data.get("elements", []))
                logger.info("Received %d elements from %s", n_elements, url)
                return data
            except requests.exceptions.Timeout:
                logger.warning("Timeout from %s (attempt %d)", url, attempt)
            except requests.exceptions.HTTPError as exc:
                logger.warning("HTTP error from %s: %s", url, exc)
            except requests.exceptions.ConnectionError as exc:
                logger.warning("Connection error from %s: %s", url, exc)
            except Exception as exc:
                logger.warning("Unexpected error from %s: %s", url, exc)

        delay = base_delay * (2 ** (attempt - 1))
        if attempt < max_retries:
            logger.info("Waiting %ds before retry %d/%d...", delay, attempt + 1, max_retries)
            time.sleep(delay)

    raise RuntimeError(
        f"All {max_retries} Overpass API attempts failed. "
        "Check connectivity or try later. "
        "If persistent, consider downloading from Geofabrik free SHP: "
        "https://download.geofabrik.de/europe/greece-latest-free.shp.zip"
    )


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------

def _way_to_polygon(geometry: list[dict]) -> Polygon | None:
    """Convert Overpass way geometry (list of {lat, lon}) to Shapely Polygon.

    Returns None for degenerate or irrecoverable geometries (these are
    logged as skipped and excluded from the output).

    The function accepts unclosed rings (first != last node) and closes them.
    Invalid geometries are repaired with make_valid(); if the result is not
    a single Polygon (e.g. multipolygon), the largest ring is kept.
    """
    if not geometry or len(geometry) < 4:
        # Need at least 3 unique points + closing node
        return None

    coords = [(pt["lon"], pt["lat"]) for pt in geometry]
    if coords[0] != coords[-1]:
        coords.append(coords[0])  # force-close the ring

    try:
        poly = Polygon(coords)
    except Exception:
        return None

    if not poly.is_valid:
        poly = make_valid(poly)

    # make_valid can return non-Polygon types; keep largest polygon component
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    elif poly.geom_type not in ("Polygon",):
        return None

    if poly.is_empty or poly.area < 1e-12:
        return None

    return poly


def _parse_elements(
    elements: list[dict],
    chunk_name: str,
) -> list[dict]:
    """Parse Overpass JSON elements into a list of row dicts.

    Only 'way' type elements are processed. Nodes, relations, and ways
    with degenerate geometry are silently skipped (count logged as warning).
    """
    rows: list[dict] = []
    skipped_type = 0
    skipped_geom = 0

    for el in elements:
        if el.get("type") != "way":
            skipped_type += 1
            continue

        poly = _way_to_polygon(el.get("geometry", []))
        if poly is None:
            skipped_geom += 1
            continue

        tags = el.get("tags", {})
        building_tag = tags.get("building", "yes")

        rows.append({
            "building_id":  f"way/{el['id']}",
            "osm_id":       el["id"],
            "osm_type":     "way",
            "source":       "osm",
            "building_tag": building_tag,
            "building_use": _classify_use(building_tag, tags),
            "name":         tags.get("name"),
            "chunk":        chunk_name,
            "geometry":     poly,
        })

    if skipped_type:
        logger.debug("Skipped %d non-way elements (nodes/relations).", skipped_type)
    if skipped_geom:
        logger.warning(
            "Chunk '%s': skipped %d way elements with invalid/degenerate geometry.",
            chunk_name, skipped_geom,
        )

    return rows


# ---------------------------------------------------------------------------
# Main download
# ---------------------------------------------------------------------------

def download_buildings(
    output_path: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> gpd.GeoDataFrame:
    """Download, parse, and save OSM buildings from all configured chunks.

    Args:
        output_path:  Destination GeoPackage path.
        cfg:          Merged pipeline config (from load_config()).
        force:        Re-download even if output_path already exists.

    Returns:
        GeoDataFrame of buildings (EPSG:4326, layer='buildings').

    Raises:
        RuntimeError: If all Overpass API attempts fail for any chunk.
        ValueError:   If zero valid buildings are parsed from the API response.
    """
    if output_path.exists() and not force:
        logger.info(
            "Output already exists: %s  (pass force=True to re-download)", output_path
        )
        return gpd.read_file(output_path, layer=_LAYER_NAME, engine="pyogrio")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_cfg = cfg["sources"]["buildings"]["primary"]
    chunks: dict[str, list[float]] = {
        name: vals["bbox"]
        for name, vals in src_cfg["download_chunks"].items()
    }
    query_timeout = src_cfg.get("query_timeout_s", 300)

    all_rows: list[dict] = []

    for chunk_name, bbox_wsen in chunks.items():
        w, s, e, n = bbox_wsen
        logger.info(
            "Downloading chunk '%s' bbox=[W=%.3f, S=%.3f, E=%.3f, N=%.3f]...",
            chunk_name, w, s, e, n,
        )
        query = _build_query(bbox_wsen, timeout=query_timeout)
        data = _fetch_overpass(query)
        elements = data.get("elements", [])

        rows = _parse_elements(elements, chunk_name)
        logger.info(
            "Chunk '%s': parsed %d valid buildings from %d elements.",
            chunk_name, len(rows), len(elements),
        )
        all_rows.extend(rows)

        # Polite delay between chunks to avoid hammering Overpass
        if chunk_name != list(chunks)[-1]:
            logger.info("Waiting 10s before next chunk...")
            time.sleep(10)

    if not all_rows:
        raise ValueError(
            "Overpass API returned 0 valid building geometries across all chunks. "
            "Check query parameters and connectivity."
        )

    gdf = gpd.GeoDataFrame(all_rows, crs="EPSG:4326")

    # Deduplicate: same building_id can appear in overlapping chunk bboxes
    n_before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["building_id"]).reset_index(drop=True)
    n_dupes = n_before - len(gdf)
    if n_dupes:
        logger.info("Removed %d duplicate buildings (overlapping chunk bboxes).", n_dupes)

    n_buildings = len(gdf)
    min_buildings = cfg["pipeline"]["run"]["min_buildings"]

    if n_buildings < min_buildings:
        logger.warning(
            "Building count (%d) is below minimum threshold (%d). "
            "Consider supplementing with Microsoft Building Footprints "
            "(see config/data_sources.yaml → buildings.fallback).",
            n_buildings, min_buildings,
        )
    else:
        logger.info("Building count OK: %d (threshold: %d)", n_buildings, min_buildings)

    logger.info("Saving %d buildings to %s ...", n_buildings, output_path)
    gdf.to_file(output_path, layer=_LAYER_NAME, driver="GPKG", engine="pyogrio")
    logger.info("Done.")

    return gdf


# ---------------------------------------------------------------------------
# Sample file for tests
# ---------------------------------------------------------------------------

def save_sample(
    gdf: gpd.GeoDataFrame,
    sample_path: Path,
    n: int = 300,
    random_state: int = 42,
) -> None:
    """Save a small random sample to data/sample/ for use in pytest fixtures.

    The sample is committed to the repo and must stay small (<1MB).
    Stratified by building_use to preserve class distribution.
    """
    if len(gdf) <= n:
        sample = gdf.copy()
    else:
        # Simple stratified sample: proportional allocation per use class
        sample = (
            gdf.groupby("building_use", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), max(1, round(n * len(g) / len(gdf)))),
                    random_state=random_state,
                )
            )
            .head(n)
            .reset_index(drop=True)
        )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_file(sample_path, layer=_LAYER_NAME, driver="GPKG", engine="pyogrio")
    logger.info("Saved %d-building sample to %s", len(sample), sample_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(force: bool = False) -> None:
    """Run the buildings acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_path = resolve_path(cfg["sources"]["buildings"]["primary"]["output_file"])
    sample_path = resolve_path(cfg["pipeline"]["paths"]["data_sample"]) / "buildings_sample.gpkg"

    gdf = download_buildings(output_path, cfg, force=force)

    if not sample_path.exists() or force:
        save_sample(gdf, sample_path)

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\nTotal buildings : {len(gdf):>10,}")
    print(f"CRS             : {gdf.crs}")
    print(f"Columns         : {list(gdf.columns)}")
    print("\nBy chunk:")
    print(gdf["chunk"].value_counts().to_string())
    print("\nBy building_use:")
    print(gdf["building_use"].value_counts().to_string())
    null_geom = gdf["geometry"].isna().sum()
    invalid_geom = (~gdf.geometry.is_valid).sum()
    print(f"\nNull geometries    : {null_geom}")
    print(f"Invalid geometries : {invalid_geom}")
    print(f"\nOutput : {output_path}")
    print(f"Sample : {sample_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OSM buildings for Attica WUI coverage area."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing output.",
    )
    args = parser.parse_args()
    main(force=args.force)
