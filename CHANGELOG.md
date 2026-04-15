# Changelog

All notable changes to **WildfireRisk-EU** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses semantic-style tags (`v2.1.0`, `v2.2.0`). Entries below are reconstructed from
`git log` commit subjects — no content has been invented. See the individual commits
for diffs.

> **Scope reminder:** this project is a research methodology demonstration. All
> validation results listed below are based on `n = 4` Leave-One-Event-Out events
> and are not statistically powered. See `README.md` → *Statistical caveat* and
> `docs/leakage_audit.md` before relying on any number in this file.

---

## [v2.2.0] — 2026-04-07

Theme: *validation-driven coverage + layer complementarity*.

### Changed
- **Building coverage** extended to the full WUI bbox (`23.4–24.2` lon), and the
  full v1 + v2 pipeline was re-run against the expanded dataset (`refactor:
  validation-driven building coverage + full v1/v2 pipeline re-run`,
  `fix: extend building coverage to full AOI (23.4-24.2 lon)`).
- **Risk map** regenerated to cover all 4 LOEO events; HTML size optimized for
  GitHub Pages hosting (`fix: show all 4 LOEO events on risk map`,
  `perf: optimize risk map HTML size for GitHub Pages`,
  `refactor: clean up risk map generation after 4-event update`).

### Fixed
- Live Demo link rendering in README (`fix: use HTML link for Live Demo to avoid
  GitHub rendering issue` and its subsequent revert).
- Removed unused imports in `risk_map.py` (`fix: remove unused imports (sys,
  pandas) in risk_map.py`).

---

## [v2.1.0] — 2026-02-25

Theme: *v2 dynamic layer + 4-event LOEO + leakage audit*.

### Added
- **Layer 2 (v2): ERA5 event-context dynamic features + LightGBM + SHAP**
  — 5 dynamic features (`wind_speed_max_24h`, `wind_dir_consistency`,
  `vpd_event_day`, `dc_antecedent_30d`, `fwi_event_day`), LightGBM binary
  classifier, TreeExplainer SHAP (`feat: ERA5 event-context layer with LightGBM
  and SHAP`).
- **LOEO expanded to 4 events**: Mati 2018, Varybobi 2021, Kalamos 2015, and
  Acharnes 2021 (`feat: multi-event validation with Varybobi 2021`,
  `feat: expand LOEO to 4 events (Kalamos 2015, Acharnes 2021)`).
- **Leakage mitigation**: fire history features excluded from LOEO feature set;
  class balancing added. Full per-event temporal cutoff still deferred — see
  `docs/leakage_audit.md` (`fix: mitigate temporal leakage in LOEO and add class
  balancing`).
- **Diagnostics**: ERA5 resolution diagnostic (showing 226,314 buildings fall
  into ~28 grid cells) and Monte Carlo weight sensitivity analysis (`feat:
  bootstrap CI, ERA5 resolution diagnostics, weight sensitivity`).
- **Documentation artifacts**: `docs/leakage_audit.md`, per-event validation
  report, comprehensive README with v2 architecture (`docs: add leakage audit
  and validation report artifacts`, `docs: comprehensive README with v2
  architecture and results`).
- **Tests & CI**: acquisition tests, CI workflow, unit tests for the scoring
  engine, validator, and QC (`test: add acquisition tests and CI workflow`,
  `test: unit tests for scoring engine, validator, and QC`).
- MIT license (`chore: add MIT license, bump to v2.1.0, final cleanup`).

### Fixed
- Makefile entrypoints and missing `pyarrow` dependency (`fix: correct Makefile
  entrypoints and add missing pyarrow dependency`).
- Ruff lint errors across the codebase (`style: fix ruff lint errors across
  codebase`).

---

## Pre-v2.1.0 — v1 baseline (no tag)

These commits predate the first tagged release and correspond to the v1
structural pipeline:

### Added
- **Mati 2018 backtest** — first v1 validation event with AUC and failure
  analysis (`feat: Mati 2018 wildfire backtest with AUC and failure analysis`).
- **Risk scoring engine** with percentile normalization and quintile binning
  (`feat: risk scoring engine with percentile normalization`).
- **Output generation**: risk map, tables, validation report, executive memo
  (`feat: output generation — risk map, tables, report, executive memo`).
- **Structural features** — terrain, vegetation, fire weather climatology,
  fire history (`feat: terrain feature extraction at building centroids`,
  `feat: vegetation and fire history feature engineering`, `feat: fire weather
  climatology features from ERA5 FWI`).
- **Preprocessing** — buildings, terrain, vegetation, FWI, fire history into
  DuckDB (`feat: building and terrain preprocessing with DuckDB`, `feat:
  vegetation, FWI, and fire history preprocessing`).
- **Data acquisition** — OSM buildings, Copernicus DEM, CORINE, EFFIS, ERA5-FWI,
  FIRMS VIIRS (`feat: OSM building acquisition and Copernicus DEM downloader`,
  `feat: CORINE, EFFIS, and ERA5-FWI data acquisition`, `feat: FIRMS VIIRS
  hotspot acquisition`).
- **Project scaffold** — config, environment, initial layout (`init: project
  scaffold with config and environment`).

---

[v2.2.0]: https://github.com/trouties/wildfire-risk-eu-pipeline/releases/tag/v2.2.0
[v2.1.0]: https://github.com/trouties/wildfire-risk-eu-pipeline/releases/tag/v2.1.0
