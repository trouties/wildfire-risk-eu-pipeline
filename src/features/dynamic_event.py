"""
src/features/dynamic_event.py
-------------------------------
Compute 5 event-context dynamic features per building for a given fire event.

Features
--------
  wind_speed_max_24h   — max wind speed (m/s) in 24 h before event end-of-day
  wind_dir_consistency — 1 / circular_std(wind direction) in 12 h window
  vpd_event_day        — max vapor pressure deficit (hPa) on event day
  dc_antecedent_30d    — mean Drought Code in 30 days before event
  fwi_event_day        — Fire Weather Index on event date

Inputs
------
  ERA5-Land hourly:  data/raw/era5_dynamic/{event_name}_hourly.nc
                     (from src/acquire/era5_dynamic.py)
  FWI daily:         data/raw/fwi/fwi_attica_daily_2015_2024.nc
                     (from src/acquire/era5_fwi.py)
  Buildings:         DuckDB table ``buildings`` (centroid_lat / centroid_lon)

Method
------
  1. Compute features per ERA5 grid cell from hourly / daily data.
  2. Assign to buildings via nearest-neighbour cKDTree on (lon, lat)
     — same pattern as ``fire_weather.py``.
  3. Write to DuckDB table ``features_dynamic_event`` with ``event_id`` column,
     using CREATE IF NOT EXISTS + DELETE/INSERT for multi-event coexistence.

Caveats
-------
  - ERA5-Land resolution is ~9 km; all buildings in a single grid cell receive
    identical dynamic feature values.
  - Wind direction consistency uses circular statistics (Mardia & Jupp 2000);
    a small epsilon (0.01 rad) prevents division-by-zero for perfectly steady wind.
  - VPD computed via Tetens / Magnus approximation (same as era5_fwi.py).
  - DC and FWI read from the pre-computed daily FWI NetCDF (Van Wagner 1987).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# Feature column names (used for consistency across functions)
FEATURE_COLS = [
    "wind_speed_max_24h",
    "wind_dir_consistency",
    "vpd_event_day",
    "dc_antecedent_30d",
    "fwi_event_day",
]


# ---------------------------------------------------------------------------
# Physical computations
# ---------------------------------------------------------------------------

def vpd_hpa(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    """Compute vapor pressure deficit (hPa) from T and Td in Kelvin.

    VPD = es(T) − ea(Td) using the Tetens / Magnus formula
    (same formulation as ``era5_fwi._rh_from_t_td``).
    """
    t_c = t_k - 273.15
    td_c = td_k - 273.15
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    ea = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    return np.maximum(es - ea, 0.0)


def circular_std(angles_rad: np.ndarray, axis: int = 0) -> np.ndarray:
    """Circular standard deviation of angles (radians).

    σ_circ = √(−2 ln R)  where R is the mean resultant length:
    R = √(mean(sin θ)² + mean(cos θ)²).

    Reference: Mardia & Jupp (2000), *Directional Statistics*.
    """
    mean_sin = np.nanmean(np.sin(angles_rad), axis=axis)
    mean_cos = np.nanmean(np.cos(angles_rad), axis=axis)
    r = np.sqrt(mean_sin**2 + mean_cos**2)
    r = np.clip(r, 1e-10, 1.0)  # avoid log(0)
    return np.sqrt(-2.0 * np.log(r))


# ---------------------------------------------------------------------------
# Grid-level feature computation
# ---------------------------------------------------------------------------

def compute_dynamic_features(
    hourly_nc: Path,
    fwi_nc: Path,
    event_date: str,
) -> pd.DataFrame:
    """Compute 5 dynamic features per ERA5 grid cell.

    Args:
        hourly_nc: ERA5 hourly NetCDF for the event window.
        fwi_nc:    Daily FWI NetCDF (2015–2024).
        event_date: "YYYY-MM-DD".

    Returns:
        DataFrame with columns: latitude, longitude, + 5 feature columns.
    """
    dt = datetime.strptime(event_date, "%Y-%m-%d")

    # ── Load hourly ERA5 ─────────────────────────────────────────────────
    ds_h = xr.open_dataset(hourly_nc)
    time_dim = "valid_time" if "valid_time" in ds_h.dims else "time"

    lats = ds_h["latitude"].values
    lons = ds_h["longitude"].values
    times = pd.DatetimeIndex(ds_h[time_dim].values)

    # Time-window masks
    event_end = pd.Timestamp(event_date) + pd.Timedelta(hours=23)
    mask_24h = (times >= event_end - pd.Timedelta(hours=23)) & (times <= event_end)
    mask_12h = (times >= event_end - pd.Timedelta(hours=11)) & (times <= event_end)
    mask_day = (times >= pd.Timestamp(event_date)) & (times <= event_end)

    u10 = ds_h["u10"].values  # (time, lat, lon), m/s
    v10 = ds_h["v10"].values
    t2m = ds_h["t2m"].values  # K
    d2m = ds_h["d2m"].values  # K
    ds_h.close()

    # ── Feature 1: wind_speed_max_24h (m/s) ──────────────────────────────
    ws = np.sqrt(u10**2 + v10**2)
    wind_max_24h = np.nanmax(ws[mask_24h], axis=0)

    # ── Feature 2: wind_dir_consistency ──────────────────────────────────
    wind_dir = np.arctan2(v10[mask_12h], u10[mask_12h])  # radians
    circ_std = circular_std(wind_dir, axis=0)
    # Higher value = more consistent direction; epsilon avoids div/0
    wind_dir_con = 1.0 / (circ_std + 0.01)

    # ── Feature 3: vpd_event_day (hPa) ──────────────────────────────────
    vpd_all = vpd_hpa(t2m[mask_day], d2m[mask_day])
    vpd_max = np.nanmax(vpd_all, axis=0)

    # ── Features 4 & 5 from existing daily FWI pipeline ──────────────────
    ds_fwi = xr.open_dataset(fwi_nc)
    fwi_time_dim = "time" if "time" in ds_fwi.dims else "valid_time"
    fwi_times = pd.DatetimeIndex(ds_fwi[fwi_time_dim].values)

    # Feature 4: dc_antecedent_30d — mean DC in [t−30d, t−1d]
    # Compare by date only — FWI timestamps may be at 12:00 UTC (noon)
    fwi_dates = fwi_times.normalize()  # strip time, keep date
    d_start = pd.Timestamp(dt - timedelta(days=30))
    d_end = pd.Timestamp(dt - timedelta(days=1))
    mask_30d = (fwi_dates >= d_start) & (fwi_dates <= d_end)
    if mask_30d.sum() == 0:
        raise ValueError(
            f"No FWI data in 30-day antecedent window [{d_start.date()}, {d_end.date()}]"
        )
    dc_mean_30d = np.nanmean(ds_fwi["dc"].values[mask_30d], axis=0)

    # Feature 5: fwi_event_day — FWI on event date
    mask_event = fwi_dates == pd.Timestamp(event_date)
    if mask_event.sum() == 0:
        raise ValueError(f"No FWI data for event date {event_date}")
    fwi_event = ds_fwi["fwi"].values[mask_event][0]

    fwi_lats = ds_fwi["latitude"].values
    fwi_lons = ds_fwi["longitude"].values
    ds_fwi.close()

    # ── Align FWI grid → hourly grid ─────────────────────────────────────
    # Both are ERA5-Land 0.1° grids; should be identical.  If not, regrid.
    if np.allclose(lats, fwi_lats, atol=0.01) and np.allclose(lons, fwi_lons, atol=0.01):
        dc_grid = dc_mean_30d
        fwi_grid = fwi_event
    else:
        from scipy.interpolate import RegularGridInterpolator

        # ERA5 latitudes may be descending; sort ascending for interpolator
        fwi_lat_sorted = fwi_lats if fwi_lats[0] < fwi_lats[-1] else fwi_lats[::-1]
        dc_sorted = dc_mean_30d if fwi_lats[0] < fwi_lats[-1] else dc_mean_30d[::-1]
        fwi_sorted = fwi_event if fwi_lats[0] < fwi_lats[-1] else fwi_event[::-1]

        interp_dc = RegularGridInterpolator(
            (fwi_lat_sorted, fwi_lons), dc_sorted,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )
        interp_fwi = RegularGridInterpolator(
            (fwi_lat_sorted, fwi_lons), fwi_sorted,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )

        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
        pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        dc_grid = interp_dc(pts).reshape(len(lats), len(lons))
        fwi_grid = interp_fwi(pts).reshape(len(lats), len(lons))

    # ── Assemble per-cell DataFrame ──────────────────────────────────────
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")

    return pd.DataFrame({
        "latitude": grid_lat.ravel(),
        "longitude": grid_lon.ravel(),
        "wind_speed_max_24h": wind_max_24h.ravel(),
        "wind_dir_consistency": wind_dir_con.ravel(),
        "vpd_event_day": vpd_max.ravel(),
        "dc_antecedent_30d": dc_grid.ravel(),
        "fwi_event_day": fwi_grid.ravel(),
    })


# ---------------------------------------------------------------------------
# Building assignment via cKDTree
# ---------------------------------------------------------------------------

def _load_building_centroids(db_path: Path) -> pd.DataFrame:
    """Return building_id, centroid_lat, centroid_lon (WGS84)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT building_id, centroid_lat, centroid_lon
        FROM buildings
        ORDER BY building_id
    """).df()
    con.close()
    return df


def assign_to_buildings(
    grid_features: pd.DataFrame,
    buildings: pd.DataFrame,
    event_id: str,
) -> pd.DataFrame:
    """Assign grid-cell features to buildings via nearest-neighbour cKDTree.

    Sea-masked cells (any NaN across the 5 features) are excluded from the
    KD-tree so every building maps to a valid land cell.
    """
    land_mask = grid_features[FEATURE_COLS].notna().all(axis=1)
    land_cells = grid_features[land_mask].reset_index(drop=True)

    if len(land_cells) == 0:
        raise ValueError("No valid land cells found in grid features.")

    # KD-tree on (lon, lat) — same order as fire_weather.py
    cell_coords = land_cells[["longitude", "latitude"]].values
    tree = cKDTree(cell_coords)

    bldg_coords = buildings[["centroid_lon", "centroid_lat"]].values
    _, nearest_idx = tree.query(bldg_coords, k=1, workers=-1)

    matched = land_cells.iloc[nearest_idx].reset_index(drop=True)

    return pd.DataFrame({
        "event_id": event_id,
        "building_id": buildings["building_id"].values,
        **{col: matched[col].values for col in FEATURE_COLS},
    })


# ---------------------------------------------------------------------------
# DuckDB write — multi-event safe (CREATE IF NOT EXISTS + DELETE/INSERT)
# ---------------------------------------------------------------------------

def _write_duckdb(df: pd.DataFrame, db_path: Path, event_id: str) -> int:
    """Write dynamic features to DuckDB for one event."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS features_dynamic_event (
            event_id                TEXT,
            building_id             TEXT,
            wind_speed_max_24h      DOUBLE,
            wind_dir_consistency    DOUBLE,
            vpd_event_day           DOUBLE,
            dc_antecedent_30d       DOUBLE,
            fwi_event_day           DOUBLE
        )
    """)
    con.execute(
        "DELETE FROM features_dynamic_event WHERE event_id = ?",
        [event_id],
    )

    con.register("_dyn_df", df)
    con.execute("""
        INSERT INTO features_dynamic_event
        SELECT event_id, building_id,
               wind_speed_max_24h, wind_dir_consistency,
               vpd_event_day, dc_antecedent_30d, fwi_event_day
        FROM _dyn_df
    """)

    n = con.execute(
        "SELECT count(*) FROM features_dynamic_event WHERE event_id = ?",
        [event_id],
    ).fetchone()[0]

    row = con.execute("""
        SELECT
            count(*)                              AS n_rows,
            round(avg(wind_speed_max_24h),  2)    AS avg_wind,
            round(max(wind_speed_max_24h),  2)    AS max_wind,
            round(avg(wind_dir_consistency), 2)   AS avg_wdc,
            round(avg(vpd_event_day),       2)    AS avg_vpd,
            round(avg(dc_antecedent_30d),   2)    AS avg_dc30,
            round(avg(fwi_event_day),       2)    AS avg_fwi
        FROM features_dynamic_event
        WHERE event_id = ?
    """, [event_id]).fetchone()

    print(f"  [duckdb] features_dynamic_event ({event_id}):")
    print(f"    rows={row[0]:,}  avg_wind={row[1]} m/s  max_wind={row[2]} m/s")
    print(f"    avg_wdc={row[3]}  avg_vpd={row[4]} hPa")
    print(f"    avg_dc30={row[5]}  avg_fwi={row[6]}")

    con.close()
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(event_name: str | None = None) -> None:
    """Compute dynamic event features for one or all validation events."""
    cfg = load_config()
    root = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]
    fwi_nc = resolve_path("data/raw/fwi/fwi_attica_daily_2015_2024.nc")

    events = cfg["validation"]["holdout_events"]
    if event_name:
        if event_name not in events:
            raise SystemExit(
                f"Unknown event: {event_name}. Available: {list(events)}"
            )
        events = {event_name: events[event_name]}

    # Load buildings once
    print("[dynamic_event] Loading building centroids ...")
    buildings = _load_building_centroids(db_path)
    print(f"  {len(buildings):,} buildings")

    for name, ev in events.items():
        print(f"\n[dynamic_event] Processing {name} ({ev['date']}) ...")

        hourly_nc = resolve_path(f"data/raw/era5_dynamic/{name}_hourly.nc")
        if not hourly_nc.exists():
            print(f"  [SKIP] Hourly data not found: {hourly_nc}")
            print(f"  Run: python -m src.acquire.era5_dynamic --event {name}")
            continue

        if not fwi_nc.exists():
            print(f"  [SKIP] FWI data not found: {fwi_nc}")
            print("  Run: python -m src.acquire.era5_fwi")
            continue

        # Step 1: compute grid-level features
        grid_df = compute_dynamic_features(
            hourly_nc=hourly_nc,
            fwi_nc=fwi_nc,
            event_date=ev["date"],
        )
        n_land = grid_df[FEATURE_COLS].notna().all(axis=1).sum()
        print(f"  Grid cells: {len(grid_df)} total, {n_land} land")
        print(
            f"  wind_max: [{grid_df['wind_speed_max_24h'].min():.1f}, "
            f"{grid_df['wind_speed_max_24h'].max():.1f}] m/s"
        )
        print(
            f"  vpd_max:  [{grid_df['vpd_event_day'].min():.1f}, "
            f"{grid_df['vpd_event_day'].max():.1f}] hPa"
        )
        print(
            f"  fwi_day:  [{grid_df['fwi_event_day'].min():.1f}, "
            f"{grid_df['fwi_event_day'].max():.1f}]"
        )

        # Step 2: assign to buildings via cKDTree
        bldg_df = assign_to_buildings(grid_df, buildings, event_id=name)

        # Step 3: write to DuckDB
        n = _write_duckdb(bldg_df, db_path, event_id=name)
        print(f"  Written {n:,} rows to features_dynamic_event for {name}")


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compute event-context dynamic features per building.",
    )
    parser.add_argument(
        "event", nargs="?", default=None,
        help="Event name (default: all validation events).",
    )
    args = parser.parse_args()
    main(event_name=args.event)
