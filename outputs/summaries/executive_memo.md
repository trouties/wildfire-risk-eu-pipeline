> **Research demonstration — NOT for underwriting use.**
> This memo summarises a research methodology demonstration. The underlying validation
> uses **n = 4 events** (Leave-One-Event-Out), with mean v2 LightGBM AUC ≈ 0.508
> (essentially chance-level); v2 is below 0.50 on 3 of 4 events. The "risk classes 1–5"
> below are **quintile bins** (each contains exactly 20% of the buildings by
> construction via `pd.qcut(q=5)` in `src/scoring/engine.py`), not absolute risk
> thresholds. At Acharnes 2021 the per-class burn rates are inverted — Class 5 burned
> at 2.2% while Class 2 burned at 3.9%. Fire-weather climatology temporal cutoffs
> are defined in config but not enforced in the feature code
> (see `docs/leakage_audit.md`). **Do not use this document for pricing, exclusion,
> sub-limit, accumulation, or any other underwriting decision.**

# Wildfire Exposure Assessment — Attica Region, Greece

**To**: *(research audience — not for underwriting)*
**From**: Geospatial Risk Analytics
**Date**: 2026-04-07
**Re**: Methods demonstration — layered wildfire scoring on 226,314 Attica WUI buildings

---

## Purpose

This memo summarises the **WildfireRisk-EU v2 research demonstration** scoring
226,314 buildings across an Attica WUI subset (dense central Athens excluded). Scores
combine a structural susceptibility layer (terrain, vegetation, fire weather
climatology, fire history) with an experimental event-context dynamic layer
(ERA5-derived wind / VPD / drought + LightGBM). **This is a research deliverable. It
is not an underwriting tool** and must not be used for re-pricing, exclusion,
sub-limit, or accumulation decisions. See the Key Limitations section below, and
`docs/leakage_audit.md` for the open-mitigation list.

---

## Portfolio Summary — quintile-binned risk classes

Each class contains exactly 20% of the 226,314-building sample **by construction**
(`pd.qcut(q=5)`). "Very High" denotes rank within this sample, not absolute
probability of loss. At Acharnes 2021, Class 5 burn rate was 2.2% vs Class 2's 3.9%
— the ranking is not calibrated. See `outputs/reports/validation_report.md`.

| Risk Class | Buildings (n) | Sample (%) | Interpretation |
|------------|--------------|------------|----------------|
| 5 — Very High | 45,263 | 20.0% | Top-quintile composite score within this sample |
| 4 — High | 45,263 | 20.0% | Fourth quintile |
| 3 — Medium | 45,262 | 20.0% | Middle quintile |
| 2 — Low | 45,263 | 20.0% | Second quintile |
| 1 — Very Low | 45,263 | 20.0% | Bottom quintile |

45,263 buildings (exactly 20.0% of the sample, by quintile construction) fall into
the top composite-score bin. With mean v2 LOEO AUC ≈ 0.508 and the Acharnes
class-rank inversion noted above, this bin is a rank-order grouping **within this
sample**, not a calibrated probability of loss.

---

## Validation Summary (n = 4 events, LOEO)

The model was backtested against 4 historical wildfire events in Attica using
leave-one-event-out cross-validation (21 features for LOEO; fire history features
excluded due to known temporal leakage). With n = 4, per-event AUCs are single
observations and no aggregate survives a standard significance test.

| Event | v1 AUC | v2 AUC |
|-------|--------|--------|
| Kalamos 2015 (30 burned) | 0.78 | 0.37 |
| Mati 2018 | 0.43 | 0.71 |
| Varybobi 2021 | 0.75 | 0.44 |
| Acharnes 2021 * | 0.50 | 0.74 |
| *Mean over 3 non-Acharnes events (post-hoc)* | *0.65* | *0.51* |

\* *Acharnes is labeled "out-of-distribution" after observing per-event results — with
n = 4 this is a post-hoc choice, not an a-priori hold-out design. The "in-distribution
mean" is therefore an average over 3 points.*

**Observation** (not a validated conclusion): across these 4 events, v1 scored
higher on Kalamos and Varybobi; v2 scored higher on Mati and Acharnes. With n = 1
per (layer × fire-type) cell, this is a description of 4 points, not a validated
boundary. v2 mean AUC ≈ 0.508 is effectively chance level, consistent with the
ERA5 9 km resolution limit (226,314 buildings into ~28 grid cells — see the
validation report's ERA5 Resolution Diagnostic).

---

## Key Limitations

1. **n = 4 events** — *"LOEO with 4 events is minimally viable but not statistically
   powerful"* (from `outputs/reports/validation_report.md`). This limitation dominates
   all others below. Expanding to 6+ events requires AOI extension beyond Attica and
   is a v3 priority.

2. **ERA5 resolution** — ~9 km grid; 226,314 buildings fall into ~28 grid cells, so
   the 5 dynamic features have no sub-cell variability. This is the structural reason
   v2 mean AUC ≈ 0.508 across events.

3. **Risk classes are uncalibrated quintile bins** — `pd.qcut(q=5)` places exactly 20%
   of buildings in each class by construction, independent of absolute risk. At
   Acharnes the class rank is inverted (Class 5 burn rate 2.2% vs Class 2 3.9%).

4. **Fire weather climatology cutoffs are not enforced** — `fwi_cutoff` and
   `fire_history_cutoff` are defined per event in `config/validation.yaml` but are
   not applied in `src/features/fire_weather.py`. `fwi_season_max` in particular is
   not diluted by averaging; a single event-day extreme enters the feature at full
   strength. See `docs/leakage_audit.md`.

5. **Proxy perimeters** — all 4 events use circular literature-proxy perimeters, not
   actual fire scars; label noise at perimeter edges.

6. **Suburban encroachment blind spot** — fires spreading from wildland into
   low-vegetation built-up areas (Acharnes archetype) are poorly separated by the
   current feature space.

---

## Research Interpretation Notes *(not underwriting guidance)*

Per-class or per-tier underwriting actions are **intentionally omitted**. With mean
v2 LOEO AUC ≈ 0.508, an inverted class-rank at Acharnes, risk classes that are
quintile bins by construction, and known-unenforced leakage cutoffs, there is no
defensible basis for tier-specific pricing, sub-limit, exclusion, or accumulation
action based on this demonstration.

If this model were to be promoted past research status, prerequisites would include
(at minimum):

1. AUC ≥ 0.75 on **≥ 6 out-of-sample events** drawn from multiple Mediterranean
   regions, not a single administrative area;
2. **Calibration curves** on the quintile bins demonstrating that Class 5 actually
   experiences a higher empirical burn rate than Class 2 across held-out events;
3. Enforcement of the per-event temporal cutoffs listed in `docs/leakage_audit.md`,
   including `fwi_season_max`;
4. A pre-registered rule for labelling events as in-distribution vs out-of-distribution,
   decided *before* seeing results.

> **Additional caveat**: For suburban WUI locations at the wildland edge (Acharnes,
> Mati coast archetype), the structural scores appear to under-rank actual exposure
> on the 4 events studied. This is a research observation on the current sample, not
> an operational rule.

---

*WildfireRisk-EU v2 | Run date: 2026-04-07 | 4-event LOEO validation*
*Data vintage: ESA WorldCover 2021, EFFIS 2000–2024, ERA5 2015–2024*
*Scores are for risk stratification only and do not constitute actuarial pricing.*
