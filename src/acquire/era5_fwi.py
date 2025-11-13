"""
src/acquire/era5_fwi.py
------------------------
Download ERA5-Land meteorological inputs and compute daily Fire Weather Index (FWI)
for the Attica AOI.

Output
------
data/raw/fwi/
  fwi_attica_daily_2015_2024.nc   (NetCDF4, daily FWI components, EPSG:4326)
  era5_land_inputs.nc             (raw ERA5-Land download, kept for reproducibility)
  manifest.json                   (provenance record)

FWI output variables
--------------------
  ffmc  Fine Fuel Moisture Code    (0–101)
  dmc   Duff Moisture Code         (0–unbounded, typical <500)
  dc    Drought Code               (0–unbounded, typical <1500)
  isi   Initial Spread Index       (0–unbounded)
  bui   Buildup Index              (0–unbounded)
  fwi   Fire Weather Index         (0–unbounded)

Primary acquisition path: ERA5-Land via Copernicus CDS
-------------------------------------------------------
  Dataset:    reanalysis-era5-land
  Variables:  2m_temperature, 2m_dewpoint_temperature,
              10m_u_component_of_wind, 10m_v_component_of_wind,
              total_precipitation
  Time:       12:00 UTC (noon; standard FWI observation time for UTC+2 area)
  Resolution: 0.1° × 0.1° (~10 km), native ERA5-Land
  Licence:    requires Copernicus CDS ERA5-Land licence (accept at
              https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)

NOTE on FWI computation
-----------------------
The original T04 plan targeted the pre-computed 'cems-fire-historical' dataset.
That dataset returns 404 on the new Copernicus CDS (post-2024 migration) and no
longer exists on the web portal. ERA5-Land with on-the-fly FWI computation is
the config-documented fallback (fallback_dataset: reanalysis-era5-land).

FWI equations: Van Wagner (1987), Canadian Forest Service Technical Report.
Implementation follows the equations exactly; no external FWI library dependency.

Precipitation handling
----------------------
ERA5-Land `total_precipitation` at 12:00 UTC equals the 0–12h accumulation
from midnight (the preceding analysis start). For Greece (UTC+2 in summer),
12:00 UTC = 14:00 local — a reasonable proxy for the standard noon observation
window. Daily rain in the dry Mediterranean summer (fire season May–September)
is near zero; any undercounting of afternoon rain affects DC/DMC minimally.
Limitation is documented in manifest.json.

Initialization
--------------
FWI codes are initialized January 1 of each year with standard over-winter
recovery values (FFMC=85, DMC=6, DC=15). A minimum 30-day spin-up period is
provided before any downstream analysis begins.

CRS / Resolution / Cadence
---------------------------
  CRS:       EPSG:4326 (native ERA5-Land)
  Grid:      0.1° × 0.1°, ~8 lat × 9 lon points over Attica AOI
  Time step: daily (one FWI value per day per grid cell)
  Calendar:  standard Gregorian
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import get_bbox, load_config, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CDS_DATASET = "reanalysis-era5-land"

# Standard FWI over-winter initialisation values (Van Wagner 1987)
_FFMC_INIT = 85.0
_DMC_INIT  =  6.0
_DC_INIT   = 15.0

# Monthly day-length adjustment factors for DMC (Le, Van Wagner 1987 Table 1)
_DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]

# Monthly drying day-length factors for DC (Lf, Van Wagner 1987 Table 2)
_DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


# ---------------------------------------------------------------------------
# CDS download
# ---------------------------------------------------------------------------

def _build_era5_request_for_month(
    cfg: dict[str, Any],
    year: int,
    month: int,
) -> dict[str, Any]:
    """Build a cdsapi request for one year-month chunk."""
    fwi_cfg = cfg["sources"]["era5_fwi"]
    area = fwi_cfg.get("area", [38.3, 23.4, 37.6, 24.2])  # [N, W, S, E]

    return {
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["12:00"],
        "area": area,
        "data_format": "netcdf",
    }

def _download_era5_inputs(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    """Download ERA5-Land meteorological inputs via CDS API in monthly chunks.

    Returns:
        Path to the merged NetCDF file (era5_land_inputs.nc).
    """
    import cdsapi
    import xarray as xr

    out_nc = output_dir / "era5_land_inputs.nc"
    if out_nc.exists() and not force:
        logger.info("ERA5-Land inputs already exist: %s", out_nc.name)
        return out_nc

    fwi_cfg = cfg["sources"]["era5_fwi"]
    date_range = fwi_cfg.get("date_range", ["2015-01-01", "2024-12-31"])
    start_year = int(date_range[0][:4])
    end_year = int(date_range[1][:4])

    client = cdsapi.Client(quiet=False)

    chunk_dir = output_dir / "_era5_chunks"
    if chunk_dir.exists() and force:
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    monthly_files: list[Path] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            chunk_nc = chunk_dir / f"era5_land_{year}_{month:02d}.nc"
            if chunk_nc.exists() and not force:
                logger.info("Chunk exists, skipping: %s", chunk_nc.name)
                monthly_files.append(chunk_nc)
                continue

            request = _build_era5_request_for_month(cfg, year, month)
            tmp_zip = chunk_dir / f"era5_land_{year}_{month:02d}.zip"

            logger.info(
                "Submitting CDS request for %04d-%02d, area=%s",
                year, month, request["area"],
            )

            client.retrieve(_CDS_DATASET, request, str(tmp_zip))

            with zipfile.ZipFile(tmp_zip, "r") as zf:
                nc_entries = [e for e in zf.namelist() if e.endswith(".nc")]
                if not nc_entries:
                    raise RuntimeError(
                        f"No .nc file in ERA5 ZIP for {year}-{month:02d}. Contents: {zf.namelist()}"
                    )
                extracted = Path(zf.extract(nc_entries[0], chunk_dir))
                shutil.move(str(extracted), str(chunk_nc))

            tmp_zip.unlink(missing_ok=True)
            monthly_files.append(chunk_nc)
            logger.info("Saved chunk: %s", chunk_nc.name)

    if not monthly_files:
        raise RuntimeError("No ERA5 monthly files were downloaded.")

    logger.info("Merging %d monthly ERA5 files ...", len(monthly_files))
    ds = xr.open_mfdataset(
        [str(p) for p in monthly_files],
        combine="by_coords",
        parallel=False,
    ).sortby("valid_time")

    ds.to_netcdf(out_nc)
    ds.close()

    logger.info("Merged ERA5 inputs saved: %s (%.1f MB)", out_nc.name,
                out_nc.stat().st_size / 1024 / 1024)
    return out_nc


# ---------------------------------------------------------------------------
# Van Wagner (1987) FWI system — vectorised over lat/lon, serial over time
# ---------------------------------------------------------------------------

def _rh_from_t_td(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    """Compute relative humidity (%) from temperature and dewpoint (Kelvin).

    Uses the Tetens / Magnus approximation (WMO standard).
    """
    t  = t_k  - 273.15   # K → °C
    td = td_k - 273.15
    # Saturation vapour pressures (hPa)
    e_s = 6.112 * np.exp(17.67 * t  / (t  + 243.5))
    e_d = 6.112 * np.exp(17.67 * td / (td + 243.5))
    return np.clip(100.0 * e_d / e_s, 0.0, 100.0)


def _wind_kmh(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert ERA5 u/v wind components (m/s) to speed in km/h."""
    return np.sqrt(u**2 + v**2) * 3.6


def _ffmc_step(m_o: np.ndarray, T: np.ndarray, H: np.ndarray,
               W: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFMC for one time step (vectorised).

    Returns:
        (ffmc, m): FFMC code and moisture content arrays.
    """
    m = m_o.copy()

    # Rain effect on FFMC
    rain_mask = r > 0.5
    if np.any(rain_mask):
        r_f = np.where(rain_mask, r - 0.5, 0.0)
        high_m = rain_mask & (m_o > 150.0)
        low_m  = rain_mask & (m_o <= 150.0)
        # Avoid log(0) and division by near-zero
        r_safe = np.where(r_f > 0, r_f, 1e-6)
        # High moisture correction
        m = np.where(
            high_m,
            m_o + 0.0015*(m_o-150.0)**2*r_f**0.5
                + 42.5*r_f*np.exp(-100.0/(251.0-m_o))*(1.0-np.exp(-6.93/r_safe)),
            m,
        )
        # Normal moisture correction
        m = np.where(
            low_m,
            m_o + 42.5*r_f*np.exp(-100.0/(251.0-m_o))*(1.0-np.exp(-6.93/r_safe)),
            m,
        )
        m = np.minimum(m, 250.0)

    # Equilibrium moisture contents
    e_d = (0.942*H**0.679 + 11.0*np.exp((H-100.0)/10.0)
           + 0.18*(21.1-T)*(1.0-np.exp(-0.115*H)))
    e_w = (0.618*H**0.753 + 10.0*np.exp((H-100.0)/10.0)
           + 0.18*(21.1-T)*(1.0-np.exp(-0.115*H)))

    # Drying
    k_d = (0.424*(1.0-(H/100.0)**1.7) + 0.0694*W**0.5*(1.0-(H/100.0)**8)
           ) * 0.581*np.exp(0.0365*T)
    # Wetting
    k_w = (0.424*(1.0-((100.0-H)/100.0)**1.7) + 0.0694*W**0.5*(1.0-((100.0-H)/100.0)**8)
           ) * 0.581*np.exp(0.0365*T)

    m_dry = np.where(m > e_d,  e_d  + (m - e_d)  * 10.0**(-k_d), m)
    m_wet = np.where(m < e_w,  e_w  - (e_w - m)  * 10.0**(-k_w), m)

    m_out = np.where(m > e_d, m_dry, np.where(m < e_w, m_wet, m))
    ffmc = np.clip(59.5 * (250.0 - m_out) / (147.2 + m_out), 0.0, 101.0)
    return ffmc, m_out


def _dmc_step(dmc_o: np.ndarray, T: np.ndarray, H: np.ndarray,
              r: np.ndarray, month: int) -> np.ndarray:
    """Compute DMC for one time step (vectorised)."""
    p = dmc_o.copy()

    # Rain effect on DMC
    rain_mask = r > 1.5
    if np.any(rain_mask):
        re = np.where(rain_mask, 0.92*r - 1.27, 0.0)
        m_o_d = 20.0 + np.exp(5.6348 - dmc_o/43.43)

        b = np.where(dmc_o <= 33.0,
                     100.0 / (0.5 + 0.3*dmc_o),
             np.where(dmc_o <= 65.0,
                     14.0 - 1.3*np.log(np.maximum(dmc_o, 1.0)),
                     6.2*np.log(np.maximum(dmc_o, 1.0)) - 17.2))

        m_r = m_o_d + 1000.0*re / (48.77 + b*re)
        arg = np.maximum(m_r - 20.0, 1.0)
        p_r = np.maximum(244.72 - 43.43*np.log(arg), 0.0)
        p = np.where(rain_mask, p_r, p)

    # Drying effect
    le = _DMC_LE[month - 1]
    k = np.where(T < -1.1, 0.0, 1.894*(T+1.1)*(100.0-H)*le*1e-6)
    return p + 100.0*k


def _dc_step(dc_o: np.ndarray, T: np.ndarray, r: np.ndarray, month: int) -> np.ndarray:
    """Compute DC for one time step (vectorised)."""
    v = dc_o.copy()

    # Rain effect on DC
    rain_mask = r > 2.8
    if np.any(rain_mask):
        rd = np.where(rain_mask, 0.83*r - 1.27, 0.0)
        q_o = 800.0*np.exp(-dc_o/400.0)
        q_r = q_o + 3.937*rd
        v_r = np.maximum(400.0*np.log(800.0/np.maximum(q_r, 1e-6)), 0.0)
        v = np.where(rain_mask, v_r, v)

    # Drying effect
    fl = _DC_LF[month - 1]
    d = np.where(T < -2.8, 0.0, np.maximum((0.36*(T+2.8) + fl)/2.0, 0.0))
    return v + d


def _isi_bui_fwi(ffmc: np.ndarray, dmc: np.ndarray, dc: np.ndarray,
                 W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ISI, BUI, FWI from FFMC, DMC, DC, wind (vectorised)."""
    # ISI
    f_W = np.exp(0.05039*W)
    m   = 147.2*(101.0-ffmc)/(59.5+ffmc)
    f_F = 91.9*np.exp(-0.1386*m)*(1.0+m**5.31/49300000.0)
    isi = 0.208*f_W*f_F

    # BUI
    bui = np.where(
        dmc <= 0.4*dc,
        0.8*dmc*dc/(dmc+0.4*dc),
        dmc - (1.0-0.8*dc/(dmc+0.4*dc))*(0.92+(0.0114*dmc)**1.7),
    )
    bui = np.maximum(bui, 0.0)

    # FWI
    f_D = np.where(bui <= 80.0,
                   0.626*bui**0.809 + 2.0,
                   1000.0/(25.0+108.64*np.exp(-0.023*bui)))
    b = 0.1*isi*f_D
    fwi = np.empty_like(b, dtype=np.float64)
    mask = b > 1.0
    fwi[~mask] = b[~mask]

    x = 0.434 * np.log(b[mask])
    fwi[mask] = np.exp(2.72 * (x ** 0.647))
    fwi = np.maximum(fwi, 0.0)

    return isi, bui, fwi


# ---------------------------------------------------------------------------
# FWI computation from ERA5 NetCDF
# ---------------------------------------------------------------------------

def _compute_fwi_from_era5(
    era5_nc: Path,
    output_dir: Path,
    cfg: dict[str, Any],
) -> Path:
    """Read ERA5-Land inputs, compute FWI system, write output NetCDF.

    Args:
        era5_nc:    Path to ERA5-Land NetCDF (t2m, d2m, u10, v10, tp).
        output_dir: Directory to write the FWI NetCDF and manifest.
        cfg:        Pipeline config.

    Returns:
        Path to the FWI output NetCDF.
    """
    import pandas as pd
    import xarray as xr

    out_nc = output_dir / "fwi_attica_daily_2015_2024.nc"

    logger.info("Loading ERA5-Land inputs: %s", era5_nc.name)
    ds = xr.open_dataset(era5_nc)
    logger.info("  dims: %s", dict(ds.sizes))
    logger.info("  vars: %s", list(ds.data_vars))

    # ── Extract and convert variables ─────────────────────────────────────────
    # ERA5 variable names
    t2m = ds["t2m"].values    # Kelvin
    d2m = ds["d2m"].values    # Kelvin
    u10 = ds["u10"].values    # m/s
    v10 = ds["v10"].values    # m/s
    tp  = ds["tp"].values     # metres (0-12h accumulation)

    # Parse time dimension
    time_vals = ds["valid_time"].values
    times = pd.DatetimeIndex(time_vals)
    lats  = ds["latitude"].values
    lons  = ds["longitude"].values

    ds.close()

    n_t, n_lat, n_lon = t2m.shape
    logger.info("Computing FWI for %d days × %d lat × %d lon ...", n_t, n_lat, n_lon)

    # Convert units
    T   = t2m - 273.15                                    # °C
    RH  = _rh_from_t_td(t2m, d2m)                        # %
    W   = _wind_kmh(u10, v10)                              # km/h
    R   = np.maximum(tp * 1000.0, 0.0)                    # mm (0-12h rain)

    # ── FWI computation ───────────────────────────────────────────────────────
    # Initialise codes with per-year startup values
    shape_2d = (n_lat, n_lon)

    # Pre-allocate output arrays
    ffmc_arr = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
    dmc_arr  = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
    dc_arr   = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
    isi_arr  = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
    bui_arr  = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
    fwi_arr  = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)

    # Running state
    ffmc_prev = np.full(shape_2d, _FFMC_INIT, dtype=np.float64)
    dmc_prev  = np.full(shape_2d, _DMC_INIT,  dtype=np.float64)
    dc_prev   = np.full(shape_2d, _DC_INIT,   dtype=np.float64)

    prev_year = times[0].year

    for t in range(n_t):
        dt    = times[t]
        month = dt.month

        # Re-initialise at start of each new year (over-winter recovery)
        if dt.year != prev_year:
            ffmc_prev[:] = _FFMC_INIT
            dmc_prev[:]  = _DMC_INIT
            dc_prev[:]   = _DC_INIT
            prev_year = dt.year

        # Convert previous FFMC to moisture content
        m_o = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

        # Compute codes
        ffmc, m_new = _ffmc_step(m_o, T[t], RH[t], W[t], R[t])
        dmc          = _dmc_step(dmc_prev, T[t], RH[t], R[t], month)
        dc           = _dc_step(dc_prev,   T[t], R[t], month)
        isi, bui, fwi = _isi_bui_fwi(ffmc, dmc, dc, W[t])

        # Store
        ffmc_arr[t] = ffmc.astype(np.float32)
        dmc_arr[t]  = dmc.astype(np.float32)
        dc_arr[t]   = dc.astype(np.float32)
        isi_arr[t]  = isi.astype(np.float32)
        bui_arr[t]  = bui.astype(np.float32)
        fwi_arr[t]  = fwi.astype(np.float32)

        # Update running state
        ffmc_prev = ffmc.astype(np.float64)
        dmc_prev  = dmc.astype(np.float64)
        dc_prev   = dc.astype(np.float64)

        if t % 365 == 0:
            logger.info("  Day %d/%d (%s) FWI sample: %.1f", t+1, n_t, dt.date(), np.nanmean(fwi))

    logger.info("FWI computation complete.")

    # ── Build output dataset ──────────────────────────────────────────────────
    attrs_common = {
        "grid_mapping": "crs",
        "coordinates": "latitude longitude",
    }

    fwi_ds = xr.Dataset(
        {
            "ffmc": xr.DataArray(ffmc_arr, dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Fine Fuel Moisture Code",
                                        "units": "1", **attrs_common}),
            "dmc":  xr.DataArray(dmc_arr,  dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Duff Moisture Code",
                                        "units": "1", **attrs_common}),
            "dc":   xr.DataArray(dc_arr,   dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Drought Code",
                                        "units": "1", **attrs_common}),
            "isi":  xr.DataArray(isi_arr,  dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Initial Spread Index",
                                        "units": "1", **attrs_common}),
            "bui":  xr.DataArray(bui_arr,  dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Buildup Index",
                                        "units": "1", **attrs_common}),
            "fwi":  xr.DataArray(fwi_arr,  dims=["time", "latitude", "longitude"],
                                 attrs={"long_name": "Fire Weather Index",
                                        "units": "1", **attrs_common}),
        },
        coords={
            "time":      ("time", times),
            "latitude":  ("latitude", lats,
                          {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", lons,
                          {"standard_name": "longitude", "units": "degrees_east"}),
        },
        attrs={
            "title": "WildfireRisk-EU daily FWI — Attica AOI 2015-2024",
            "source": "ERA5-Land (Copernicus CDS), Van Wagner (1987) FWI equations",
            "institution": "ECMWF / Copernicus Climate Change Service",
            "crs": "EPSG:4326",
            "history": f"Computed {datetime.now(timezone.utc).isoformat()}",
            "comment": (
                "FWI computed from ERA5-Land 2m_temperature, 2m_dewpoint, 10m wind, "
                "total_precipitation at 12:00 UTC. Precipitation represents 0-12h "
                "accumulation from midnight (proxy for daily rain during fire season). "
                "Codes re-initialised January 1 each year (FFMC=85, DMC=6, DC=15)."
            ),
        },
    )

    encoding = {
        var: {"dtype": "float32", "zlib": True, "complevel": 4}
        for var in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
    }
    encoding["time"] = {"dtype": "int64", "units": "hours since 1970-01-01"}

    fwi_ds.to_netcdf(out_nc, encoding=encoding)
    logger.info("FWI NetCDF written: %s (%.1f MB)", out_nc.name,
                out_nc.stat().st_size / 1024 / 1024)
    return out_nc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_fwi(
    output_dir: Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    """Download ERA5-Land inputs and compute daily FWI for the Attica AOI.

    This function implements the config-documented fallback path
    (fallback_dataset: reanalysis-era5-land) after the primary source
    'cems-fire-historical' was found to be unavailable on the new CDS API.

    Args:
        output_dir: Target directory (created if missing).
        cfg:        Merged pipeline config (from load_config()).
        force:      Re-download and recompute even if outputs already exist.

    Returns:
        Path to the FWI NetCDF file.

    Raises:
        RuntimeError: If ERA5-Land is inaccessible (licence not accepted).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_nc = output_dir / "fwi_attica_daily_2015_2024.nc"

    if out_nc.exists() and not force:
        size_mb = out_nc.stat().st_size / 1024 / 1024
        logger.info(
            "FWI output already exists: %s (%.1f MB). Pass force=True to recompute.",
            out_nc.name, size_mb,
        )
        return out_nc

    # ── Download ERA5-Land inputs ─────────────────────────────────────────────
    era5_nc = _download_era5_inputs(output_dir, cfg, force=force)

    # ── Compute FWI ───────────────────────────────────────────────────────────
    logger.info("Computing FWI from ERA5-Land inputs ...")
    out_nc = _compute_fwi_from_era5(era5_nc, output_dir, cfg)

    # ── Verify ────────────────────────────────────────────────────────────────
    import pandas as pd
    import xarray as xr

    with xr.open_dataset(out_nc) as ds:
        t_min = pd.Timestamp(ds["time"].values[0]).strftime("%Y-%m-%d")
        t_max = pd.Timestamp(ds["time"].values[-1]).strftime("%Y-%m-%d")
        stats = {
            "variables": list(ds.data_vars),
            "n_timesteps": int(ds.sizes["time"]),
            "time_range": [t_min, t_max],
            "lat_range": [float(ds["latitude"].min()), float(ds["latitude"].max())],
            "lon_range": [float(ds["longitude"].min()), float(ds["longitude"].max())],
            "fwi_summer_mean": float(
                ds["fwi"].sel(time=ds["time"].dt.month.isin([6, 7, 8, 9])).mean()
            ),
            "fwi_max": float(ds["fwi"].max()),
            "size_mb": round(out_nc.stat().st_size / 1024 / 1024, 2),
        }

    logger.info(
        "FWI verified: %d days (%s → %s), summer mean FWI=%.1f, max FWI=%.1f",
        stats["n_timesteps"], t_min, t_max,
        stats["fwi_summer_mean"], stats["fwi_max"],
    )

    bbox = get_bbox(cfg)

    # ── Write manifest ────────────────────────────────────────────────────────
    fwi_cfg = cfg["sources"]["era5_fwi"]
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "ERA5-Land (Copernicus CDS reanalysis-era5-land)",
        "provider": "ECMWF / Copernicus Climate Change Service",
        "dataset_used": _CDS_DATASET,
        "original_target_dataset": "cems-fire-historical",
        "substitution_note": (
            "'cems-fire-historical' not found on new Copernicus CDS API or web portal "
            "(HTTP 404, as of 2026-04). ERA5-Land (fallback_dataset in config) used "
            "with on-the-fly FWI computation using Van Wagner (1987) equations."
        ),
        "fwi_reference": "Van Wagner (1987), Canadian Forest Service Technical Report",
        "fwi_variables": ["ffmc", "dmc", "dc", "isi", "bui", "fwi"],
        "era5_inputs": ["2m_temperature", "2m_dewpoint_temperature",
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "total_precipitation"],
        "era5_time": "12:00 UTC (noon; FWI standard observation window)",
        "precipitation_note": (
            "total_precipitation at 12:00 UTC = 0-12h accumulation from midnight. "
            "Used as proxy for daily rain; underestimates afternoon rainfall. "
            "Acceptable for Mediterranean summer fire season context."
        ),
        "initialisation": {
            "schedule": "January 1 of each year",
            "ffmc": _FFMC_INIT, "dmc": _DMC_INIT, "dc": _DC_INIT,
        },
        "crs": "EPSG:4326",
        "resolution_deg": 0.1,
        "aoi_bbox_wsen": bbox,
        "date_range": fwi_cfg.get("date_range", ["2015-01-01", "2024-12-31"]),
        "acquisition_method": "automated (ERA5-Land CDS API + Van Wagner FWI computation)",
        "file_stats": stats,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest written: %s", manifest_path)

    return out_nc


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(force: bool = False) -> None:
    """Run FWI acquisition as a standalone script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    output_dir = resolve_path("data/raw/fwi")

    try:
        out_nc = download_fwi(output_dir, cfg, force=force)
        import xarray as xr
        with xr.open_dataset(out_nc) as ds:
            print(f"\nOutput:    {out_nc}")
            print(f"Size:      {out_nc.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"Variables: {list(ds.data_vars)}")
            print(f"Time:      {ds.sizes['time']} days")
            print(f"Grid:      {ds.sizes['latitude']}×{ds.sizes['longitude']}")
            print("Total NaNs in fwi:", int(ds["fwi"].isnull().sum()))
            print("NaN cells on first day:", int(ds["fwi"].isel(time=0).isnull().sum()))
            print("Finite cells on first day:", int(ds["fwi"].isel(time=0).notnull().sum()))
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download ERA5-Land and compute FWI for Attica AOI."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download and recompute even if outputs exist.")
    args = parser.parse_args()
    main(force=args.force)
