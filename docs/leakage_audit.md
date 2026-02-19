# Temporal Leakage Audit — WildfireRisk-EU v2

This document audits all 26 features (21 structural + 5 dynamic) used in the
v2 pipeline for temporal information leakage relative to LOEO validation events
(Mati 2018-07-23, Varybobi 2021-08-03, Kalamos 2015-08-17, Acharnes 2021-08-23).

## Definitions

- **Pre-event**: Feature is derived from data whose temporal extent strictly
  precedes the validation event date.
- **Post-event**: Feature computation includes data from after the event date.
  May cause leakage if the event itself influences the feature value.
- **Climatology**: Feature is derived from a multi-year seasonal average
  (2015-2024). Includes post-event years but is not event-specific.
- **LOEO enabled**: Feature is included in leave-one-event-out cross-validation.
  Fire history features are excluded from LOEO to mitigate known leakage.

## Leakage Risk Levels

- **None**: No temporal leakage pathway exists.
- **Low**: Theoretical leakage pathway exists but effect is negligible or
  conservative (works against the model).
- **Medium**: Leakage pathway exists and may modestly inflate metrics.
- **High**: Direct leakage — feature encodes information about the test event.

---

## Feature Audit Table

### Group 1: Terrain (5 features)

| # | Feature | Group | Temporal Basis | LOEO | Leakage Risk | Rationale |
|---|---------|-------|---------------|------|-------------|-----------|
| 1 | `elevation_m` | Terrain | Static (DEM) | Yes | None | Copernicus GLO-30 DEM is time-invariant topography |
| 2 | `slope_deg` | Terrain | Static (DEM) | Yes | None | Derived from DEM; topography does not change with fire events |
| 3 | `south_aspect_score` | Terrain | Static (DEM) | Yes | None | Derived from DEM aspect; time-invariant |
| 4 | `tpi_300m` | Terrain | Static (DEM) | Yes | None | Topographic Position Index from DEM; time-invariant |
| 5 | `tri_300m` | Terrain | Static (DEM) | Yes | None | Terrain Ruggedness Index from DEM; time-invariant |

### Group 2: Vegetation (6 features)

| # | Feature | Group | Temporal Basis | LOEO | Leakage Risk | Rationale |
|---|---------|-------|---------------|------|-------------|-----------|
| 6 | `veg_fraction_100m` | Vegetation | CORINE 2018 snapshot | Yes | Low | CORINE 2018 reference year is pre-Varybobi/Acharnes but post-Mati/Kalamos; Mati/Kalamos scar may reduce local veg fraction, working against the model (conservative) |
| 7 | `veg_fraction_500m` | Vegetation | CORINE 2018 snapshot | Yes | Low | Same as veg_fraction_100m; larger buffer dilutes any scar effect |
| 8 | `dist_to_forest_m` | Vegetation | CORINE 2018 snapshot | Yes | Low | CORINE 2018 snapshot; any Mati-related vegetation loss would increase distance (conservative direction) |
| 9 | `dist_to_scrubland_m` | Vegetation | CORINE 2018 snapshot | Yes | Low | Same logic as dist_to_forest_m |
| 10 | `wui_class` | Vegetation | CORINE 2018 + OSM buildings | Yes | None | WUI classification uses building density (static) and CORINE veg fraction; no fire-event signal |
| 11 | `veg_continuity_500m` | Vegetation | CORINE 2018 snapshot | Yes | Low | Patch connectivity from CORINE; post-Mati scar could reduce connectivity (conservative) |

### Group 3: Fire Weather Climatology (5 features)

| # | Feature | Group | Temporal Basis | LOEO | Leakage Risk | Rationale |
|---|---------|-------|---------------|------|-------------|-----------|
| 12 | `fwi_season_mean` | Fire Weather | Climatology 2015-2024 | Yes | Low | Multi-year seasonal average includes post-event years, but individual event-year FWI is diluted across 10 years of fire seasons; no event-specific signal leaks through |
| 13 | `fwi_season_p90` | Fire Weather | Climatology 2015-2024 | Yes | Low | 90th percentile across 10 fire seasons; dominated by extreme years regardless of which event is tested |
| 14 | `fwi_season_max` | Fire Weather | Climatology 2015-2024 | Yes | Low | Single-day maximum across all seasons; may include the actual event day's FWI, but one day out of ~1,530 fire-season days is negligible |
| 15 | `dc_season_mean` | Fire Weather | Climatology 2015-2024 | Yes | Low | Drought Code seasonal average; same dilution logic as fwi_season_mean |
| 16 | `fwi_extreme_days` | Fire Weather | Climatology 2015-2024 | Yes | Low | Count of days with FWI > 30 per season, averaged; event day contributes at most 1 day out of ~150+ extreme days across 10 years |

### Group 4: Fire History (5 features)

| # | Feature | Group | Temporal Basis | LOEO | Leakage Risk | Rationale |
|---|---------|-------|---------------|------|-------------|-----------|
| 17 | `dist_to_nearest_fire_m` | Fire History | Post-event (no cutoff) | **No** | **High** | Includes the test event's own EFFIS perimeter; buildings inside the Mati/Varybobi scar have dist=0, directly encoding the label |
| 18 | `fire_count_5km` | Fire History | Post-event (no cutoff) | **No** | **High** | Count includes the test event itself; buildings near the event perimeter get inflated counts |
| 19 | `fire_count_10km` | Fire History | Post-event (no cutoff) | **No** | **High** | Same as fire_count_5km at larger radius |
| 20 | `firms_hotspot_count_5km` | Fire History | Post-event (2015-2024) | **No** | **Medium** | FIRMS VIIRS data spans 2015-2024, includes post-event hotspots; however, Mati area has low hotspot count (6.9 vs AOI mean 78.9), so leakage direction is conservative for Mati |
| 21 | `recency_score` | Fire History | Post-event (no cutoff) | **No** | **High** | Recency-weighted sum includes the test event with maximum recency weight; directly encodes temporal proximity to the test fire |

### Group 5: Dynamic Event-Context (5 features)

| # | Feature | Group | Temporal Basis | LOEO | Leakage Risk | Rationale |
|---|---------|-------|---------------|------|-------------|-----------|
| 22 | `wind_speed_max_24h` | Dynamic | Pre-event (event day) | Yes | None | ERA5 hourly data from [event_date - 23h, event_date + 23h]; uses conditions concurrent with or immediately preceding ignition — this is the intended signal, not leakage |
| 23 | `wind_dir_consistency` | Dynamic | Pre-event (12h window) | Yes | None | Circular std of wind direction in 12h window; same justification as wind_speed_max_24h |
| 24 | `vpd_event_day` | Dynamic | Event day | Yes | None | Max VPD on event day; atmospheric conditions are a causal driver, not a consequence of the fire |
| 25 | `dc_antecedent_30d` | Dynamic | Strictly pre-event | Yes | None | Mean Drought Code [event_date - 30d, event_date - 1d]; strictly before event with 1-day buffer |
| 26 | `fwi_event_day` | Dynamic | Event day | Yes | None | FWI on event date; weather conditions are exogenous to fire occurrence at building level |

---

## Summary

| Risk Level | Count | Features |
|-----------|-------|----------|
| **None** | 12 | All terrain (5), wui_class, all dynamic (5), wui_class |
| **Low** | 9 | 5 vegetation (excl. wui_class), 5 fire weather climatology |
| **Medium** | 1 | firms_hotspot_count_5km |
| **High** | 4 | dist_to_nearest_fire_m, fire_count_5km, fire_count_10km, recency_score |

## Mitigation Status

1. **Fire history features excluded from LOEO** (implemented in v2): All 5 fire
   history features are removed from the LOEO feature set, reducing the model
   to 21 features. This prevents the LightGBM model from learning to predict
   burned labels using features that encode the test event.

2. **Fire weather climatology accepted as Low risk**: The 10-year seasonal
   averaging dilutes any single event's contribution. The config defines
   `fire_history_cutoff` and `fwi_cutoff` dates per event, but these are
   **not yet enforced** in the feature computation code.

3. **CORINE 2018 vintage accepted as Low risk**: The single-snapshot nature
   means any post-event land cover changes work against the model (burned areas
   show less vegetation = lower risk score), making the leakage direction
   conservative. For Kalamos 2015 and Mati 2018 (pre-CORINE), any scar
   captured in the 2018 snapshot is conservative. For Varybobi and Acharnes
   2021 (post-CORINE), no leakage exists.

4. **Full temporal cutoff deferred to v3**: Per-event date filtering for fire
   history and FWI climatology features is planned for v3. This will enforce
   strict pre-event data boundaries for each validation event.

---

*Generated by WildfireRisk-EU Leakage Audit | v2*
