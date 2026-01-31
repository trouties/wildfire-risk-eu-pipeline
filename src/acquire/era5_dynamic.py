"""
src/acquire/era5_dynamic.py
----------------------------
Download ERA5-Land hourly meteorological inputs for event-context dynamic
features around a specific fire event date.

Output
------
data/raw/era5_dynamic/
  {event_name}_hourly.nc       (NetCDF4, hourly u10/v10/t2m/d2m, EPSG:4326)
  {event_name}_manifest.json   (provenance record)

Variables
---------
  u10  10m u-component of wind  (m/s)
  v10  10m v-component of wind  (m/s)
  t2m  2m temperature           (K)
  d2m  2m dewpoint temperature  (K)

Acquisition
-----------
  Dataset:    reanalysis-era5-land
  Time:       all 24 hours per day
  Days:       [event_date − 1, event_date]  (48 h window)
  Resolution: 0.1° × 0.1° (~10 km), native ERA5-Land
  Area:       Attica AOI (same as era5_fwi.py config)
  Licence:    Copernicus CDS ERA5-Land (same as existing FWI pipeline)

The 48 h window provides:
  - Full 24 h pre-event window for wind_speed_max_24h
  - 12 h pre-event window for wind_dir_consistency
  - All event-day hours for vpd_event_day
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)

_CDS_DATASET = "reanalysis-era5-land"

_HOURLY_TIMES = [f"{h:02d}:00" for h in range(24)]

_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
]


# ---------------------------------------------------------------------------
# CDS request builder
# ---------------------------------------------------------------------------

def _build_requests(
    event_date: datetime,
    area: list[float],
) -> list[dict[str, Any]]:
    """Build CDS API request(s) for 48 h window [event_date − 1, event_date].

    Returns one request if both days share the same year-month,
    otherwise two separate requests (month-boundary case).
    """
    d0 = event_date - timedelta(days=1)
    d1 = event_date

    base = {
        "variable": _VARIABLES,
        "time": _HOURLY_TIMES,
        "area": area,
        "data_format": "netcdf",
    }

    if d0.year == d1.year and d0.month == d1.month:
        return [{
            **base,
            "year": [str(d1.year)],
            "month": [f"{d1.month:02d}"],
            "day": [f"{d0.day:02d}", f"{d1.day:02d}"],
        }]

    # Month boundary — two separate requests
    return [
        {
            **base,
            "year": [str(d0.year)],
            "month": [f"{d0.month:02d}"],
            "day": [f"{d0.day:02d}"],
        },
        {
            **base,
            "year": [str(d1.year)],
            "month": [f"{d1.month:02d}"],
            "day": [f"{d1.day:02d}"],
        },
    ]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _extract_nc_from_zip(zip_path: Path, dest: Path) -> Path:
    """Extract the first .nc file from a CDS ZIP download."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        nc_entries = [e for e in zf.namelist() if e.endswith(".nc")]
        if not nc_entries:
            raise RuntimeError(
                f"No .nc file in ERA5 ZIP. Contents: {zf.namelist()}"
            )
        extracted = Path(zf.extract(nc_entries[0], dest.parent))
        shutil.move(str(extracted), str(dest))
    zip_path.unlink(missing_ok=True)
    return dest


def download_era5_hourly(
    event_name: str,
    event_date: str,
    output_dir: Path | None = None,
    cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Download ERA5-Land hourly data for a 48 h event window.

    Args:
        event_name: Event identifier (e.g. "mati_2018").
        event_date: Event date string "YYYY-MM-DD".
        output_dir: Target directory (default: data/raw/era5_dynamic).
        cfg:        Pipeline config (loaded if None).
        force:      Re-download even if output exists.

    Returns:
        Path to the merged hourly NetCDF file.
    """
    import cdsapi
    import xarray as xr

    if cfg is None:
        cfg = load_config()
    if output_dir is None:
        output_dir = resolve_path("data/raw/era5_dynamic")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_nc = output_dir / f"{event_name}_hourly.nc"

    if out_nc.exists() and not force:
        logger.info("ERA5 hourly data exists: %s", out_nc.name)
        return out_nc

    dt = datetime.strptime(event_date, "%Y-%m-%d")
    area = cfg["sources"]["era5_fwi"].get("area", [38.3, 23.4, 37.6, 24.2])
    requests = _build_requests(dt, area)

    client = cdsapi.Client(quiet=False)
    chunk_paths: list[Path] = []

    for i, req in enumerate(requests):
        chunk_nc = output_dir / f"_chunk_{event_name}_{i}.nc"
        tmp_zip = chunk_nc.with_suffix(".zip")

        logger.info(
            "CDS request %d/%d for %s: year=%s month=%s day=%s",
            i + 1, len(requests), event_name,
            req["year"], req["month"], req["day"],
        )
        client.retrieve(_CDS_DATASET, req, str(tmp_zip))
        _extract_nc_from_zip(tmp_zip, chunk_nc)
        chunk_paths.append(chunk_nc)

    # Merge chunks if needed
    if len(chunk_paths) == 1:
        shutil.move(str(chunk_paths[0]), str(out_nc))
    else:
        ds = xr.open_mfdataset(
            [str(p) for p in chunk_paths],
            combine="by_coords",
        ).sortby("valid_time")
        ds.to_netcdf(out_nc)
        ds.close()
        for p in chunk_paths:
            p.unlink(missing_ok=True)

    # Verify
    with xr.open_dataset(out_nc) as ds:
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        n_times = ds.sizes[time_dim]
        var_list = list(ds.data_vars)
        logger.info(
            "ERA5 hourly saved: %s — %d timesteps, vars=%s, %.1f MB",
            out_nc.name, n_times, var_list,
            out_nc.stat().st_size / 1024 / 1024,
        )

    # Write manifest
    d0 = dt - timedelta(days=1)
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "event_name": event_name,
        "event_date": event_date,
        "source": f"ERA5-Land (Copernicus CDS {_CDS_DATASET})",
        "variables": _VARIABLES,
        "temporal_resolution": "hourly",
        "window": f"{d0.date()} to {dt.date()}",
        "area_nswe": area,
        "crs": "EPSG:4326",
        "resolution_deg": 0.1,
    }
    manifest_path = output_dir / f"{event_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest: %s", manifest_path)

    return out_nc


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Download ERA5 hourly for all (or one) validation events."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download ERA5-Land hourly data for event-context features.",
    )
    parser.add_argument(
        "--event", type=str, default=None,
        help="Single event name (default: all validation events).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if outputs exist.",
    )
    args = parser.parse_args()

    cfg = load_config()
    events = cfg["validation"]["holdout_events"]

    if args.event:
        if args.event not in events:
            raise SystemExit(f"Unknown event: {args.event}. Available: {list(events)}")
        events = {args.event: events[args.event]}

    for name, ev in events.items():
        out = download_era5_hourly(name, ev["date"], cfg=cfg, force=args.force)
        print(f"  {name}: {out}")


if __name__ == "__main__":
    main()
