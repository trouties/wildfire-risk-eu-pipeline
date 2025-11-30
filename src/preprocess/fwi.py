"""
preprocess/fwi.py — Normalize FWI NetCDF and compute per-grid-cell seasonal statistics.

FWI is computed on a coarse 0.1° grid (8 lat × 9 lon) already clipped to the Attica
AOI.  T08 will bilinearly interpolate the seasonal stats to individual building
centroids.

Inputs:
  data/raw/fwi/fwi_attica_daily_2015_2024.nc   (ERA5-Land + Van Wagner FWI, 2015–2024)

Outputs:
  data/processed/fwi_daily_attica.nc            daily FWI (all variables, standardized path)
  data/processed/fwi_season_stats_attica.nc     per-cell fire-season statistics (2015–2024)
  data/wildfire_risk.duckdb → fwi_grid_stats    flat stats table for DuckDB queries

Seasonal stats computed per grid cell over all fire-season days (June–October):
  fwi_season_mean        mean daily FWI
  fwi_season_p90         90th-percentile daily FWI
  fwi_season_max         maximum daily FWI
  dc_season_mean         mean daily Drought Code
  fwi_extreme_days       mean days/season with FWI > 30
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fire_season_mask(times: np.ndarray, start_month: int = 6, end_month: int = 10
                      ) -> np.ndarray:
    """Return boolean array: True for timesteps within the fire season months."""
    months = pd.DatetimeIndex(times).month
    return (months >= start_month) & (months <= end_month)


def _compute_season_stats(ds_season: xr.Dataset, extreme_threshold: float = 30.0
                          ) -> xr.Dataset:
    """Compute per-cell summary statistics over all fire-season timesteps."""
    fwi = ds_season["fwi"]
    dc  = ds_season["dc"]

    fwi_mean = fwi.mean(dim="time")
    fwi_p90  = fwi.quantile(0.9, dim="time").drop_vars("quantile")
    fwi_max  = fwi.max(dim="time")
    dc_mean  = dc.mean(dim="time")

    # Mean extreme-FWI days per year (averaged across 2015–2024 fire seasons)
    is_extreme = fwi > extreme_threshold           # bool DataArray
    extreme_by_year = (
        is_extreme
        .groupby(is_extreme.time.dt.year)
        .sum()                                     # sum per year per cell
    )
    # extreme_by_year has dim 'year'; average across years
    extreme_mean = extreme_by_year.mean(dim="year")

    ds_stats = xr.Dataset(
        {
            "fwi_season_mean":   fwi_mean.rename("fwi_season_mean"),
            "fwi_season_p90":    fwi_p90.rename("fwi_season_p90"),
            "fwi_season_max":    fwi_max.rename("fwi_season_max"),
            "dc_season_mean":    dc_mean.rename("dc_season_mean"),
            "fwi_extreme_days":  extreme_mean.rename("fwi_extreme_days"),
        }
    )
    return ds_stats


def _load_stats_to_duckdb(ds_stats: xr.Dataset, db_path: Path) -> None:
    """Flatten the stats DataArrays into a per-cell table in DuckDB."""
    lats = ds_stats.latitude.values
    lons = ds_stats.longitude.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape (n_lat, n_lon)

    def _safe(val: float) -> float | None:
        """Convert NaN to None so DuckDB stores SQL NULL, not a float NaN."""
        return None if (val is None or np.isnan(val)) else val

    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cell_id = f"LAT{lat:.1f}_LON{lon:.1f}".replace(".", "p")
            rows.append({
                "cell_id":           cell_id,
                "latitude":          float(lat),
                "longitude":         float(lon),
                "fwi_season_mean":   _safe(float(ds_stats["fwi_season_mean"].values[i, j])),
                "fwi_season_p90":    _safe(float(ds_stats["fwi_season_p90"].values[i, j])),
                "fwi_season_max":    _safe(float(ds_stats["fwi_season_max"].values[i, j])),
                "dc_season_mean":    _safe(float(ds_stats["dc_season_mean"].values[i, j])),
                "fwi_extreme_days":  _safe(float(ds_stats["fwi_extreme_days"].values[i, j])),
            })

    df = pd.DataFrame(rows)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("DROP TABLE IF EXISTS fwi_grid_stats;")
    con.register("_fwi_df", df)
    con.execute("""
        CREATE TABLE fwi_grid_stats AS
        SELECT
            cell_id,
            latitude,
            longitude,
            fwi_season_mean,
            fwi_season_p90,
            fwi_season_max,
            dc_season_mean,
            fwi_extreme_days
        FROM _fwi_df
    """)

    count    = con.execute("SELECT count(*) FROM fwi_grid_stats").fetchone()[0]
    max_fwi  = con.execute("SELECT max(fwi_season_max) FROM fwi_grid_stats").fetchone()[0]
    print(f"  [duckdb] fwi_grid_stats: {count} cells | peak fwi_season_max = {max_fwi:.1f}")
    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg  = load_config()
    root = resolve_path(".")

    raw_path   = root / "data/raw/fwi/fwi_attica_daily_2015_2024.nc"
    proc_daily = root / "data/processed/fwi_daily_attica.nc"           # expected by features.yaml
    proc_stats = root / "data/processed/fwi_season_stats_attica.nc"
    db_path    = root / cfg["pipeline"]["paths"]["db"]

    proc_daily.parent.mkdir(parents=True, exist_ok=True)

    fire_season_start = cfg["pipeline"]["run"]["fire_season_start"]   # 6
    fire_season_end   = cfg["pipeline"]["run"]["fire_season_end"]     # 10
    extreme_threshold = (
        cfg["features"]["fire_weather"]["features"]["fwi_extreme_days"]["extreme_threshold"]
    )

    # ------------------------------------------------------------------
    # 1. Read raw NetCDF
    # ------------------------------------------------------------------
    print(f"[fwi] Reading {raw_path.name} ...")
    ds = xr.open_dataset(raw_path)
    print(f"  dims     : {dict(ds.dims)}")
    print(f"  variables: {list(ds.data_vars)}")
    print(f"  time     : {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")
    print(f"  lat      : {ds.latitude.values[[0, -1]]}  lon: {ds.longitude.values[[0, -1]]}")

    # ------------------------------------------------------------------
    # 2. Save standardized daily copy
    # ------------------------------------------------------------------
    print(f"[fwi] Saving daily copy → {proc_daily.name} ...")
    ds.to_netcdf(proc_daily)
    print(f"  size: {proc_daily.stat().st_size / 1e6:.2f} MB")

    # ------------------------------------------------------------------
    # 3. Compute fire-season statistics
    # ------------------------------------------------------------------
    print(f"[fwi] Filtering to fire season (months {fire_season_start}–{fire_season_end}) ...")
    season_mask = _fire_season_mask(
        ds.time.values, fire_season_start, fire_season_end
    )
    ds_season = ds.isel(time=season_mask)
    n_days_season = int(season_mask.sum())
    print(f"  fire-season days: {n_days_season}  ({n_days_season}/{len(ds.time)} total)")

    print(f"[fwi] Computing seasonal statistics (extreme threshold FWI > {extreme_threshold}) ...")
    ds_stats = _compute_season_stats(ds_season, extreme_threshold=extreme_threshold)

    # ------------------------------------------------------------------
    # 4. Save stats NetCDF
    # ------------------------------------------------------------------
    print(f"[fwi] Saving seasonal stats → {proc_stats.name} ...")
    ds_stats.to_netcdf(proc_stats)

    # QC print (use nanmean/nanmin/nanmax so ocean cells with NaN don't hide land stats)
    print("\n[fwi] Seasonal stats summary (mean across valid grid cells):")
    for var in ds_stats.data_vars:
        vals = ds_stats[var].values.astype(float)
        n_valid = int(np.isfinite(vals).sum())
        print(f"  {var:<24}: mean={np.nanmean(vals):.2f}  "
              f"min={np.nanmin(vals):.2f}  max={np.nanmax(vals):.2f}  "
              f"(valid cells: {n_valid}/{vals.size})")

    # ------------------------------------------------------------------
    # 5. Load into DuckDB
    # ------------------------------------------------------------------
    print(f"[fwi] Loading fwi_grid_stats into DuckDB → {db_path.name} ...")
    _load_stats_to_duckdb(ds_stats, db_path)

    ds.close()
    print("\n[fwi] Done.")


if __name__ == "__main__":
    main()
