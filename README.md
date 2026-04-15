# WildfireRisk-EU: Layered Wildfire Risk Intelligence for Mediterranean WUI

[![CI](https://github.com/trouties/wildfire-risk-eu-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/trouties/wildfire-risk-eu-pipeline/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![DuckDB](https://img.shields.io/badge/storage-DuckDB-orange.svg)](https://duckdb.org/)

A reproducible geospatial data pipeline that scores **226,314 buildings** in the Athens wildland-urban interface (WUI) for wildfire risk using a **two-layer architecture** — structural susceptibility plus event-context dynamics — evaluated against **four historical fires** via leave-one-event-out cross-validation. With n=4, results constitute a **methodological demonstration**, not a statistically powered validation or a production-ready underwriting model. See the [Statistical caveat](#statistical-caveat) and [Limitations](#limitations) before reading the numbers below.

> **[Live Demo: Interactive Risk Map](https://trouties.github.io/wildfire-risk-eu-pipeline/)** — 226K buildings color-coded by risk class on an interactive Folium map

| | |
|---|---|
| **Stakeholder** | Re/insurance portfolio underwriter assessing European wildfire exposure |
| **AOI** | Attica WUI subset — effective bbox `[23.4, 37.6, 24.2, 38.3]`, WUI-filtered to 226,314 buildings (dense central Athens excluded); working CRS EPSG:2100. The *administrative* Attica region is ~3,808 km²; this project covers the WUI bbox, not the full admin area. |
| **Validation** | Leave-one-event-out against 4 Attica fires (Kalamos 2015, Mati 2018, Varybobi 2021, Acharnes 2021) |
| **Key Outputs** | [Executive Memo](outputs/summaries/executive_memo.md) · [Validation Report](outputs/reports/validation_report.md) · [Interactive Map](https://trouties.github.io/wildfire-risk-eu-pipeline/) |

---

## v2 Architecture — a methodological experiment

> **v2 is an experiment, not a validated deliverable.** It asks: *can ERA5-derived event-day meteorology (~9 km grid) add building-level discrimination beyond v1 structural features?* Across 4 events the result is largely a **negative finding** — mean LOEO AUC ≈ 0.508, effectively chance-level, driven by the structural fact that 226,314 buildings fall into only ~28 ERA5 grid cells (see [ERA5 resolution diagnostic](outputs/reports/validation_report.md#era5-resolution-diagnostic)). The 5 dynamic features therefore cannot distinguish neighboring buildings; they only distinguish between grid cells, which in a cross-event setting collapses to encoding event identity. The architecture below documents what was *built*; the [Validation Results](#validation-results) document what it *does*.

```
Layer 1: Structural Susceptibility (v1)
  "This building is in a long-run high-risk zone"
  → terrain, vegetation, fire weather climatology, fire history
  → 21 features → weighted composite index

Layer 2: Event-Context Dynamic (v2) — experimental
  "Under today's conditions, this building's risk is amplified/dampened"
  → ERA5 hourly wind speed/direction, VPD, antecedent drought, event-day FWI
  → 5 features → LightGBM binary classifier → SHAP explainability
  → (building-level signal limited by ~9 km ERA5 grid; see caveats below)

Combined: Impact-Priority Score
  → structural × event-context → prioritized risk ranking
```

### Pipeline Stages

Dependency chain matches `make all-v2` (`Makefile:93`): v1 validation runs *before* v2 LightGBM.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1a    │    │  Stage 2     │    │  Stage 3a    │    │  Stage 4     │
│  Acquire     │───▶│  Preprocess  │───▶│  Structural  │───▶│  Scoring     │
│  (structural)│    │              │    │  Features    │    │  (v1 index)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
┌──────────────┐    ┌──────────────┐                               ▼
│  Stage 1b    │    │  Stage 3b    │                       ┌──────────────┐
│  Acquire ERA5│───▶│  Dynamic     │──────────────────────▶│  Stage 5     │
│  (hourly)    │    │  Features    │                       │  v1 Validate │
└──────────────┘    └──────────────┘                       │  (LOEO×4)    │
                                                           └──────┬───────┘
                                                                  │
                                                                  ▼
                                                           ┌──────────────┐
                                                           │  Stage 5b    │
                                                           │  v2 LightGBM │
                                                           │  + SHAP      │
                                                           └──────┬───────┘
                                                                  │
                                                                  ▼
                                                           ┌──────────────┐
                                                           │  Stage 6     │
                                                           │  Outputs     │
                                                           └──────────────┘
                                              DuckDB storage
                                           (data/wildfire_risk.duckdb)
```

---

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate geo311

# 2. Install package in editable mode
pip install -e ".[dev]"

# 3. Run full v2 pipeline (structural + event-context + model)
make all-v2

# Or run v1 structural pipeline only:
make all

# Or run stages individually:
make acquire            # Stage 1a: structural data sources
make acquire-dynamic    # Stage 1b: ERA5 hourly event-context data
make preprocess         # Stage 2:  clean, harmonize, load to DuckDB
make features           # Stage 3a: structural per-building features
make features-dynamic   # Stage 3b: event-context dynamic features
make score              # Stage 4:  v1 weighted composite index
make validate           # Stage 5:  backtest (4 LOEO events)
make model              # Stage 5b: LightGBM + SHAP
make outputs            # Stage 6:  deliverables (reports, maps, tables)
```

---

## Data Sources

| Source | Provider | Resolution | License |
|--------|----------|------------|---------|
| OSM Buildings | OpenStreetMap | Building-level | ODbL |
| Copernicus GLO-30 DEM | Copernicus | 30m | Free |
| CORINE Land Cover 2018 | CLMS | 100m | Free |
| EFFIS Fire Perimeters | JRC | Vector | Free |
| FWI (ERA5-Land daily) | CDS/ECMWF | ~9km | Free |
| ERA5-Land hourly (v2) | CDS/ECMWF | ~9km | Free |
| NDVI HR-VPP | CLMS | 10m | Free |
| FIRMS VIIRS | NASA | 375m | Free |

See `config/data_sources.yaml` for download endpoints and the [Credentials Required](#credentials-required) section below.

---

## Features

### Layer 1: Structural Features (21 effective, 4 groups)

| Group | Weight\* | Features |
|-------|---------|---------|
| Vegetation | 33% | veg\_fraction (100m/500m), dist\_to\_forest, dist\_to\_scrubland, wui\_class, veg\_continuity |
| Terrain | 22% | elevation, slope, south\_aspect\_score, TPI, TRI |
| Fire Weather | 22% | FWI mean/p90/max, Drought Code, extreme-FWI days |
| Fire History | 22% | nearest fire distance, fire counts 5/10km, FIRMS hotspots, recency score |

\*Proximity weight (10%) redistributed proportionally to the 4 present groups at runtime.

### Layer 2: Event-Context Dynamic Features (5, v2)

| Feature | Source | Description |
|---------|--------|-------------|
| `wind_speed_max_24h` | ERA5 hourly u10/v10 | Max wind speed (m/s) in 24h pre-event window |
| `wind_dir_consistency` | ERA5 hourly u10/v10 | 1 / circular\_std of wind direction (12h window) |
| `vpd_event_day` | ERA5 hourly t2m/d2m | Max vapor pressure deficit (hPa) on event day |
| `dc_antecedent_30d` | FWI daily | Mean Drought Code in 30 days before event |
| `fwi_event_day` | FWI daily | FWI value on event date |

---

## Validation Results

### Statistical caveat

> **Read this before the numbers.** Validation uses **only n=4 events**. With 4 samples the per-event AUCs are single-observation readings, not a sampling distribution; the "in-distribution mean" reported below is computed after *post-hoc* removal of one event (Acharnes) and therefore rests on **3 points**. No metric in this section survives a standard statistical-significance test, and no single 0.70 "threshold" should be read as pass/fail. Treat this entire section as a **method demonstration**, not as evidence of deployability. Kalamos 2015 has only 30 burned buildings (CI [0.69, 0.86]); any strong-sounding generalization from that point is unwarranted.

### 4 Events × 2 Layers Matrix (Leave-One-Event-Out)

| Event | Fire type | v1 Structural AUC [95% CI] | v2 LOEO AUC [95% CI] | Δ AUC |
|-------|-----------|---------------------------|----------------------|-------|
| Kalamos 2015 | Terrain-driven | 0.779 [0.69, 0.86] | 0.373 [0.27, 0.47] | −0.406 |
| Mati 2018 | Wind-driven | 0.431 [0.42, 0.45] | 0.715 [0.70, 0.73] | +0.283 |
| Varybobi 2021 | Terrain-driven | 0.748 [0.74, 0.76] | 0.435 [0.42, 0.45] | −0.313 |
| Acharnes 2021 † | Suburban encroachment | 0.502 [0.49, 0.52] | 0.737 [0.73, 0.74] | +0.235 |
| *Mean over 3 non-Acharnes events (post-hoc)* | | *0.653* | *0.508* | *−0.145* |

† Acharnes is labeled out-of-distribution *after* observing per-event results. With n=4 this is a **post-hoc choice**, not an a-priori hold-out design; the "in-distribution mean" is an average over the remaining 3 points and should be treated accordingly. The v2 mean ≈ 0.508 is effectively chance-level.

### Per-event observations (each row is n=1 — do not generalize)

- **Kalamos 2015** — v1 AUC 0.779 with only 30 burned buildings; the 95% CI [0.69, 0.86] is wide. v2 AUC 0.373 is below chance. A single very small event cannot support a "v1 is good on terrain fires" claim.
- **Mati 2018** — v2 AUC 0.715 on a single wind-driven event; v1 AUC 0.431 is below chance here. Interesting as a case study, but n=1 — do not generalize to "v2 is better on wind-driven fires."
- **Varybobi 2021** — v1 AUC 0.748 exceeds the 0.70 reference, v2 AUC 0.435. Again, one event.
- **Acharnes 2021** — v1 AUC 0.502, v2 AUC 0.737. Note: at Acharnes the per-class burn rate is **inverted** — Class 5 ("Very High") has a 2.2% burn rate while Class 2 ("Low") has 3.9% (see `outputs/reports/validation_report.md`). The AUC number masks this miscalibration.
- **Layer "complementarity"** — with n=1 per (layer × fire-type) cell, the pattern ("v1 wins terrain, v2 wins wind/suburban") is a *description* of these 4 points, not a validated boundary. Fire-type-aware model selection is a v3 research direction, not an established result.
- **Leakage status** — fire-history features are excluded from LOEO. **FWI / fire-weather climatology cutoffs are defined in `config/validation.yaml` but not yet enforced in `src/features/fire_weather.py`** — in particular `fwi_season_max` is not diluted by averaging, so a single event-day extreme enters the feature at full strength. The project is *not* claimed to be leakage-free; see [docs/leakage_audit.md](docs/leakage_audit.md) for the open-mitigation list.

### ROC Curves (Leave-One-Event-Out)

<p align="center">
  <img src="outputs/reports/kalamos_2015_roc_curve.png" alt="Kalamos 2015 ROC" width="24%">
  <img src="outputs/reports/mati_2018_roc_curve.png" alt="Mati 2018 ROC" width="24%">
  <img src="outputs/reports/varybobi_2021_roc_curve.png" alt="Varybobi 2021 ROC" width="24%">
  <img src="outputs/reports/acharnes_2021_roc_curve.png" alt="Acharnes 2021 ROC" width="24%">
</p>

### Observed per-event results (n = 1 each — descriptive, not a boundary claim)

| Fire type (n=1) | v1 AUC | v2 AUC | Observation |
|-----------------|--------|--------|-------------|
| Terrain/fuel-driven (Kalamos) | 0.779 | 0.373 | v1 higher on this single event; v2 below chance. CI wide (30 burned buildings). |
| Wind-driven (Mati)            | 0.431 | 0.715 | v2 higher on this single event; v1 below chance. |
| Terrain-driven large (Varybobi)| 0.748 | 0.435 | v1 higher on this single event; v2 below chance. |
| Suburban encroachment (Acharnes)| 0.502 | 0.737 | v2 higher on this single event; class ranking is inverted (Class 5 burn rate 2.2% vs Class 2 3.9%). |

With one observation per cell, no column of this table is a "boundary" in any statistical sense. The pattern is a description of 4 events, not evidence that either layer generalizes to unseen fires of the same type.

### Weight Sensitivity Analysis

Monte Carlo perturbation (±20%) of structural layer weights confirms score stability:

<p align="center">
  <img src="outputs/validation/weight_sensitivity.png" alt="Weight Sensitivity Analysis" width="60%">
</p>

### Structural Layer Detail (Mati 2018)

| Metric | Model | Baseline (dist-to-forest) |
|--------|-------|----------|
| Full-bbox AUC-ROC | 0.43 | 0.60 |
| South sub-zone AUC-ROC | 0.79 | 0.55 |
| Recall (risk classes 4+5) | 96% | -- |

---

## SHAP Explainability

> **Read this first.** With 226,314 buildings falling into only ~28 ERA5 grid cells, every dynamic-feature SHAP value reflects event-day weather *at the cell level*, not at the building level. Two neighboring buildings inside the same grid cell receive identical wind/VPD/FWI inputs, so SHAP cannot assign differential attribution between them. Treat the plots below as a sanity check on LightGBM's internal logic, not as a causal building-level attribution.

v2 includes SHAP (SHapley Additive exPlanations) via TreeExplainer for the LightGBM model, providing:

- **Global feature importance** — which features drive burned/unburned discrimination across all buildings
- **Beeswarm plot** — per-feature SHAP value distributions showing direction and magnitude of impact

**Top SHAP features** (mean |SHAP|, full-data model with all 26 features):
1. `dist_to_nearest_fire_m` (1.31) — proximity to historical fires
2. `elevation_m` (0.90) — terrain elevation
3. `firms_hotspot_count_5km` (0.66) — satellite hotspot density

Note: Fire history features dominate the full-data SHAP model but are **excluded from LOEO evaluation** due to temporal leakage (see `docs/leakage_audit.md`). Dynamic features rank lower because the ERA5 9 km grid limits their between-building variability — consistent with the overall v2 negative finding described in [v2 Architecture](#v2-architecture--a-methodological-experiment).

<p align="center">
  <img src="outputs/validation/shap_importance.png" alt="SHAP Feature Importance" width="48%">
  <img src="outputs/validation/shap_beeswarm.png" alt="SHAP Beeswarm Plot" width="48%">
</p>

---

## Outputs

| Deliverable | Path | Description |
|-------------|------|-------------|
| Risk table (CSV) | `outputs/tables/risk_scores_attica.csv` | Per-building v1 scores + risk class (generated, not committed) |
| Risk table (Parquet) | `outputs/tables/risk_scores_attica.parquet` | Same, Parquet format (generated, not committed) |
| Risk map | [Live Demo](https://trouties.github.io/wildfire-risk-eu-pipeline/) | Interactive Folium map — 226K buildings color-coded by risk class |
| Validation report | [`outputs/reports/validation_report.md`](outputs/reports/validation_report.md) | Multi-event AUC-ROC, lift, failure analysis |
| Executive memo | [`outputs/summaries/executive_memo.md`](outputs/summaries/executive_memo.md) | 1-page underwriter summary |
| v2 model metrics | [`outputs/validation/v2_model_metrics.json`](outputs/validation/v2_model_metrics.json) | LightGBM LOEO AUC per event |
| SHAP importance | [`shap_importance.png`](outputs/validation/shap_importance.png) | Global feature importance bar chart |
| SHAP beeswarm | [`shap_beeswarm.png`](outputs/validation/shap_beeswarm.png) | Per-feature SHAP value distribution |
| v2 predictions | DuckDB `model_v2_predictions` | Per-building LightGBM probabilities per event |

---

## Project Structure

```
wildfire-risk-eu/
├── config/            # Pipeline configuration (YAML)
├── src/
│   ├── acquire/       # Stage 1: data download (structural + ERA5 hourly)
│   ├── preprocess/    # Stage 2: cleaning, CRS harmonization
│   ├── features/      # Stage 3: structural + dynamic feature extraction
│   ├── scoring/       # Stage 4: v1 normalization + weighted index
│   ├── validation/    # Stage 5: multi-event backtest
│   ├── model/         # Stage 5b: LightGBM event-context model + SHAP (v2)
│   ├── outputs/       # Stage 6: deliverable generation
│   ├── qc/            # Quality control checks
│   └── utils/         # Shared config loader + DuckDB schema
├── tests/             # pytest test suite (9 files, 214 tests)
├── data/
│   ├── raw/           # Downloaded raw data (gitignored)
│   ├── processed/     # Processed intermediates (gitignored)
│   └── sample/        # Small test data subset (committed)
├── docs/              # Leakage audit, executive memo
└── outputs/           # Generated deliverables (gitignored except sample)
```

---

## Engineering

- **Tests**: 9 test files, 214 tests covering acquisition, scoring engine, validation, QC, dynamic features, and event model
- **CI**: GitHub Actions — ruff lint + pytest on every push/PR
- **Storage**: DuckDB with 9 analytical tables (7 structural + 2 v2)
- **Config**: 5 YAML files driving all pipeline parameters
- **Reproducibility**: Full pipeline from `make all-v2` with documented credential setup

---

## Credentials Required

| Service | Location |
|---------|----------|
| CDS API (ERA5 download) | `~/.cdsapirc` |
| Copernicus DEM | Windows Credential Manager: `CopernicusDataSpace` |
| NASA Earthdata | Windows Credential Manager: `EarthdataToken` |
| NASA FIRMS | Environment variable: `MAP_KEY` |

---

## Limitations

- **Sample size (n = 4 events)** — *"LOEO with 4 events is minimally viable but not statistically powerful"* (from `outputs/reports/validation_report.md`). With 4 observations, any per-event AUC is a single reading, the in-distribution vs out-of-distribution partition is inevitably post-hoc, and no comparison of means survives a statistical-significance test. Expanding to 6+ events requires AOI extension beyond Attica and is a v3 priority. **This limitation should be read as dominating all others below.**
- **ERA5 resolution** — ~9 km grid; 226,314 buildings fall into ~28 cells. The v2 dynamic features therefore have no sub-cell variability and in a cross-event LOEO setting collapse to encoding event identity. This is the structural reason v2 mean AUC ≈ 0.508.
- **Fire weather climatology cutoffs not enforced** — `config/validation.yaml` defines `fwi_cutoff` and `fire_history_cutoff` per event, but `src/features/fire_weather.py` does not apply them. `fwi_season_max` in particular is *not* diluted by averaging — a single event-day extreme enters the feature at full strength — so its leakage pathway is stronger than the mean-based climatology features. Full temporal cutoff enforcement is deferred to v3. See [`docs/leakage_audit.md`](docs/leakage_audit.md).
- **Fire history leakage** — fire history features include post-event data and are *excluded* from LOEO; full per-event temporal cutoff is deferred to v3.
- **Proxy perimeters** — all 4 events use circular literature-proxy perimeters (area-equivalent radius from EFFIS annual reports), not actual fire boundaries. The `fallback_source` entries in config (EMSR300, EMSR531) are documentation-only and not wired into the validator.
- **Fire-type description, not boundary** — v1 scored higher on Kalamos and Varybobi; v2 scored higher on Mati and Acharnes. With n=1 per (layer × fire-type) cell, this is a description of 4 events, not a validated "applicability boundary."
- **Risk classes are quintile bins** — `src/scoring/engine.py` uses `pd.qcut(q=5)`, so each class contains exactly 20% of the portfolio by construction. "Very High" denotes rank within this sample, not an absolute probability of loss. At Acharnes the per-class burn rates are inverted (Class 5: 2.2% vs Class 2: 3.9%).
- **CORINE 2018 vintage** — post-2018 land-use changes not captured.
- **AOI limited to Attica WUI subset** — generalization to other Mediterranean WUI zones is untested.

---

## Future Work (v3)

- Fire-type-aware LOEO or OOD event exclusion from training to prevent cross-contamination
- Suburban encroachment features (building density gradient, wind-geometry interaction)
- Per-event temporal cutoff enforcement for fire history and FWI features
- Additional validation events (6+, requires AOI expansion beyond Attica)
- Real EMS perimeters (EMSR300/EMSR531) to replace circular proxy boundaries
- Sentinel-2 NDVI/NDWI satellite features for fuel moisture state

---

*WildfireRisk-EU v2.2.0 | MIT License | [CHANGELOG](CHANGELOG.md)*
