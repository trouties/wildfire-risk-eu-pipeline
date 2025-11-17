"""
src/acquire/firms.py
---------------------
Download FIRMS VIIRS I-Band 375m active fire detections for the Attica AOI.

Output
------
data/raw/firms/
  viirs_attica_2015_2024.csv   (VIIRS SNPP standard product, 2015-01-01 to 2024-12-31)
  manifest.json                (provenance record)

Source
------
NASA FIRMS (Fire Information for Resource Management System)
  - Product: VIIRS_SNPP_SP (Suomi-NPP VIIRS Standard Product, science quality archive)
  - Spatial resolution: I-Band 375 m
  - Temporal coverage: 2012-01-20 to present
  - API endpoint: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{product}/{area}/{days}/{date}
  - API key from MAP_KEY environment variable (never hardcode)
  - Maximum day range per request: 5 days (VIIRS SP constraint; NRT allows up to 10)

Note on product selection
-------------------------
The config nominates VIIRS_SNPP_NRT but NRT covers only the last ~60 days.
For the 2015–2024 historical window, VIIRS_SNPP_SP (Standard/archive product)
is used instead. The SP product is the science-quality version; column schema
is identical to NRT.

Columns in output CSV
---------------------
latitude, longitude, bright_ti4, scan, track, acq_date, acq_time,
satellite, instrument, confidence, version, bright_ti5, frp, daynight
  - latitude / longitude: fire pixel centroid (EPSG:4326)
  - acq_date:  UTC acquisition date (YYYY-MM-DD)
  - frp:       Fire Radiative Power (MW) — proxy for fire intensity
  - confidence: detection confidence (nominal/low/high)

CRS / Resolution / Cadence notes
---------------------------------
- CRS: EPSG:4326 (WGS84)
- Pixel footprint: ~375 m × 375 m at nadir
- Repeat cycle: ~1–2 overpasses/day (day + night combined)
- Latency: SP product typically 2–5 months behind real-time

Caveats
-------
- Active fire detections are thermal anomalies, not burned-area extent.
  They complement EFFIS perimeters rather than replacing them.
- Cloud cover and smoke suppress detections; gaps are not zero-fire periods.
- Confidence filter: include 'nominal' and 'high' confidence pixels (downstream T06).
- FIRMS API limits requests to 5-day windows for SP product.
  Downloading 2015–2024 requires ~731 API calls (≈ 10–15 minutes with rate limiting).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from src.utils.config import get_bbox, load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Use standard product (science quality archive) for historical window.
# NRT product only covers the last ~60 days.
_VIIRS_PRODUCT = "VIIRS_SNPP_SP"

# VIIRS SP endpoint restricts to 5 days per request.
_MAX_DAYS_PER_REQUEST = 5

# Polite delay between API requests (seconds) to avoid rate limiting.
# 0.1s gives ~172 req/min, well within FIRMS API limits for a small AOI.
_REQUEST_DELAY_S = 0.1

# Retry settings for transient failures.
_MAX_RETRIES = 3
_RETRY_DELAY_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key(key_env: str) -> str:
    """Retrieve the FIRMS API key from the named environment variable.

    Args:
        key_env: Name of the environment variable (e.g. 'MAP_KEY').

    Returns:
        API key string.

    Raises:
        RuntimeError: If the environment variable is not set.
    """
    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(
            f"FIRMS API key not found: environment variable '{key_env}' is not set. "
            "Set it with: $env:MAP_KEY = '<your-key>'  (PowerShell) "
            "or export MAP_KEY=<your-key>  (bash)."
        )
    return key


def _bbox_to_firms_area(bbox: list[float]) -> str:
    """Convert [W, S, E, N] bbox to FIRMS area string 'W,S,E,N'."""
    west, south, east, north = bbox
    return f"{west},{south},{east},{north}"


def _date_batches(
    start: date,
    end: date,
    batch_days: int,
) -> list[tuple[date, int]]:
    """Generate (batch_start_date, day_count) pairs covering [start, end].

    Each batch covers at most `batch_days` days.

    Args:
        start:      First date to include.
        end:        Last date to include (inclusive).
        batch_days: Maximum days per batch.

    Returns:
        List of (batch_start, day_count) tuples.
    """
    batches: list[tuple[date, int]] = []
    current = start
    while current <= end:
        remaining = (end - current).days + 1
        count = min(remaining, batch_days)
        batches.append((current, count))
        current += timedelta(days=count)
    return batches


def _fetch_batch(
    api_key: str,
    product: str,
    area: str,
    batch_date: date,
    day_count: int,
    session: requests.Session,
) -> str | None:
    """Download one date batch from the FIRMS area CSV API.

    Args:
        api_key:    FIRMS API key.
        product:    VIIRS product identifier (e.g. 'VIIRS_SNPP_SP').
        area:       Area string 'W,S,E,N'.
        batch_date: Start date for this batch.
        day_count:  Number of days (1–5 for SP).
        session:    Requests session (for connection reuse).

    Returns:
        Raw CSV text (may be header-only if no detections), or None on failure.
    """
    date_str = batch_date.strftime("%Y-%m-%d")
    url = f"{_FIRMS_API_BASE}/{api_key}/{product}/{area}/{day_count}/{date_str}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            else:
                logger.warning(
                    "Batch %s: HTTP %d (attempt %d/%d): %s",
                    date_str, resp.status_code, attempt, _MAX_RETRIES,
                    resp.text[:100],
                )
        except requests.RequestException as exc:
            logger.warning(
                "Batch %s: network error (attempt %d/%d): %s",
                date_str, attempt, _MAX_RETRIES, exc,
            )

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_S)

    logger.error("Batch %s: all %d attempts failed.", date_str, _MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_firms(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    """Download FIRMS VIIRS active fire detections for the Attica AOI.

    Downloads VIIRS_SNPP_SP (standard product) in 5-day batches covering
    the date range defined in cfg['sources']['firms']['date_range'].

    Args:
        output_dir: Target directory (created if missing).
        cfg:        Merged pipeline config (from load_config()).
        force:      Re-download even if output file already exists.

    Returns:
        Path to the output CSV file.

    Raises:
        RuntimeError: If the API key is missing or all batches fail.
    """
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "viirs_attica_2015_2024.csv"
    manifest_path = output_dir / "manifest.json"

    if out_csv.exists() and not force:
        row_count = sum(1 for _ in open(out_csv, encoding="utf-8")) - 1
        logger.info(
            "FIRMS output already exists: %s (%d data rows). "
            "Pass force=True to re-download.",
            out_csv.name, max(row_count, 0),
        )
        return out_csv

    # ── Config ────────────────────────────────────────────────────────────────
    firms_cfg = cfg["sources"]["firms"]
    key_env = firms_cfg.get("api_key_env", "MAP_KEY")
    api_key = _get_api_key(key_env)

    date_range = firms_cfg.get("date_range", ["2015-01-01", "2024-12-31"])
    start_date = date.fromisoformat(date_range[0])
    end_date   = date.fromisoformat(date_range[1])

    bbox = get_bbox(cfg)  # [W, S, E, N]
    area = _bbox_to_firms_area(bbox)

    logger.info(
        "FIRMS download: %s | area=%s | %s → %s",
        _VIIRS_PRODUCT, area, start_date, end_date,
    )

    # ── Build batch list ──────────────────────────────────────────────────────
    batches = _date_batches(start_date, end_date, _MAX_DAYS_PER_REQUEST)
    total_batches = len(batches)
    logger.info("Total batches: %d (max %d days each)", total_batches, _MAX_DAYS_PER_REQUEST)

    # ── Download loop (incremental write) ────────────────────────────────────
    # Write rows directly to disk as each batch arrives.
    # This avoids losing data if the process is interrupted mid-download.
    header: str | None = None
    failed_batches: list[str] = []
    total_rows_written = 0

    session = requests.Session()
    session.headers["User-Agent"] = "WildfireRiskEU/T04 (research)"

    # Open in write mode initially; we'll append after writing the header.
    csv_fh = open(out_csv, "w", encoding="utf-8", newline="")
    try:
        for i, (batch_date, day_count) in enumerate(batches):
            if i % 50 == 0:
                logger.info(
                    "Progress: batch %d/%d (%s) — %d rows so far",
                    i + 1, total_batches,
                    batch_date.strftime("%Y-%m-%d"),
                    total_rows_written,
                )

            raw = _fetch_batch(api_key, _VIIRS_PRODUCT, area, batch_date, day_count, session)
            if raw is None:
                failed_batches.append(batch_date.strftime("%Y-%m-%d"))
                continue

            lines = raw.strip().splitlines()
            if not lines:
                continue

            # First line is the header; write once at the top of the file.
            if header is None:
                header = lines[0]
                csv_fh.write(header + "\n")
                data_lines = lines[1:]
            else:
                if lines[0] != header:
                    logger.warning(
                        "Batch %s: header mismatch — skipping data rows.",
                        batch_date.strftime("%Y-%m-%d"),
                    )
                    continue
                data_lines = lines[1:]

            for row in data_lines:
                csv_fh.write(row + "\n")
            total_rows_written += len(data_lines)

            # Polite rate-limit delay (skip on last batch)
            if i < total_batches - 1:
                time.sleep(_REQUEST_DELAY_S)

    finally:
        csv_fh.flush()
        csv_fh.close()

    if header is None:
        raise RuntimeError(
            "FIRMS download failed: no data returned from any batch. "
            "Check MAP_KEY validity and network connectivity."
        )

    logger.info("Download complete: %d rows written to %s", total_rows_written, out_csv.name)

    # ── Verify ────────────────────────────────────────────────────────────────
    import csv

    row_count = 0
    col_names: list[str] = []
    with open(out_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        col_names = reader.fieldnames or []
        for _ in reader:
            row_count += 1

    required_cols = {"latitude", "longitude", "acq_date"}
    missing = required_cols - set(col_names)
    if missing:
        raise RuntimeError(
            f"FIRMS CSV missing required columns: {missing}. "
            f"Columns found: {col_names}"
        )

    logger.info(
        "FIRMS verified: %d fire detections, columns=%s",
        row_count, col_names,
    )

    if failed_batches:
        logger.warning(
            "%d batches failed (data gap): %s ... (total %d)",
            len(failed_batches), failed_batches[:5], len(failed_batches),
        )

    # ── Write manifest ────────────────────────────────────────────────────────
    import json

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "NASA FIRMS VIIRS Suomi-NPP Active Fire (Standard Product)",
        "product": _VIIRS_PRODUCT,
        "provider": "NASA / LANCE-FIRMS",
        "api_endpoint": _FIRMS_API_BASE,
        "api_key_env": key_env,
        "spatial_resolution_m": 375,
        "crs": "EPSG:4326",
        "aoi_bbox_wsen": bbox,
        "date_range": [start_date.isoformat(), end_date.isoformat()],
        "total_batches": total_batches,
        "failed_batches": len(failed_batches),
        "failed_batch_dates": failed_batches[:20],
        "rows_downloaded": row_count,
        "columns": col_names,
        "acquisition_method": "automated (FIRMS area CSV API, 5-day batches)",
        "note": (
            "Config nominates VIIRS_SNPP_NRT but NRT covers only the last ~60 days. "
            "VIIRS_SNPP_SP (Standard Product, science quality archive) is used for the "
            "2015-2024 historical window. Column schema is identical."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written: %s", manifest_path)
    logger.info("FIRMS download complete: %d rows in %s", row_count, out_csv)

    return out_csv


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(force: bool = False) -> None:
    """Run FIRMS acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_dir = resolve_path("data/raw/firms")

    out_csv = download_firms(output_dir, cfg, force=force)

    import csv
    with open(out_csv, newline="", encoding="utf-8") as fh:
        row_count = sum(1 for _ in csv.reader(fh)) - 1  # exclude header

    print(f"\nOutput:  {out_csv}")
    print(f"Rows:    {row_count:,} fire detections")
    print(f"Size:    {out_csv.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download FIRMS VIIRS active fire detections for Attica AOI."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file exists.")
    args = parser.parse_args()
    main(force=args.force)
