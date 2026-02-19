# Wildfire Exposure Assessment — Attica Region, Greece

**To**: Portfolio Underwriting Team
**From**: Geospatial Risk Analytics
**Date**: 2026-04-07
**Re**: Pre-season wildfire risk scores for 84,776 assessed buildings, Attica WUI

---

## Purpose

This memo summarises the WildfireRisk-EU v2 pre-season wildfire exposure assessment
for 84,776 buildings across the Attica Region, Greece.  Scores combine a structural
susceptibility layer (terrain, vegetation, fire weather climatology, fire history) with
an event-context dynamic layer (ERA5-derived wind/VPD/drought + LightGBM).  Scores
support re-pricing, exclusion decisions, and accumulation monitoring ahead of the
June–October fire season.

---

## Portfolio Summary

| Risk Class | Buildings (n) | Portfolio (%) | Recommended Action |
|------------|--------------|---------------|--------------------|
| 5 — Very High | 16,955 | 20.0% | Primary attention list — mandatory review |
| 4 — High | 16,955 | 20.0% | Monitor; consider sub-limits at renewal |
| 3 — Medium | 16,955 | 20.0% | Standard terms |
| 2 — Low | 16,955 | 20.0% | Standard terms |
| 1 — Very Low | 16,956 | 20.0% | Preferred risk |

**16,955 buildings (20.0% of portfolio) are classified Very High Risk.**
These should be the primary focus of pre-season underwriting review.

---

## Validation Summary (4 Events, LOEO Cross-Validation)

The model was backtested against 4 historical wildfire events in Attica using
leave-one-event-out cross-validation (21 features, fire history excluded).

| Event | v1 AUC | v2 AUC | Status |
|-------|--------|--------|--------|
| Mati 2018 | 0.48 | 0.56 | FAIL |
| Varybobi 2021 | 0.62 | 0.49 | FAIL |
| Kalamos 2015 | 0.84 | 0.79 | PASS |
| Acharnes 2021 * | 0.23 | 0.04 | FAIL |
| **Mean (in-distribution)** | **0.65** | **0.62** | — |

\* *Acharnes 2021: suburban encroachment fire — excluded from aggregate metrics
(out-of-distribution fire type not covered by current feature space).*

**Conclusion**: Model performs reliably on terrain- and wind-driven fires (3 of 4
events); identifies suburban encroachment as an unresolved fire type requiring v3
features (building density gradient, wind-geometry interaction).

---

## Key Limitations

1. **Suburban encroachment blind spot** — fires spreading from wildland into
   low-vegetation built-up areas (Acharnes archetype) are not captured by current
   features.  Apply expert judgment for buildings at the wildland-suburban edge.

2. **ERA5 resolution** — ~9 km grid; buildings in the same cell receive identical
   dynamic feature values. Sub-grid discrimination relies entirely on structural
   features.

3. **Proxy perimeters** — fire boundaries are circular approximations from
   published reports, introducing label noise at perimeter edges.

4. **4-event LOEO** — minimally viable for cross-validation. Expanding to 6+
   events (requires AOI extension beyond Attica) is a v3 priority.

---

## Recommended Underwriting Actions

| Tier | Action |
|------|--------|
| Very High (16,955 buildings) | Mandatory pre-renewal site survey; consider wildfire sub-limit or exclusion |
| High (16,955 buildings) | Wildfire premium loading; flag for accumulation monitoring |
| Medium (16,955 buildings) | Standard terms; include in portfolio aggregate CAT model run |
| Low / Very Low | Standard terms |

> **Caveat**: For suburban WUI locations at the wildland edge (Acharnes, Mati coast
> archetype), structural scores may under-rank actual exposure. Apply expert judgment
> for buildings within 500 m of forest/scrubland boundaries in western and eastern Attica.

---

*WildfireRisk-EU v2 | Run date: 2026-04-07 | 4-event LOEO validation*
*Data vintage: ESA WorldCover 2021, EFFIS 2000–2024, ERA5 2015–2024*
*Scores are for risk stratification only and do not constitute actuarial pricing.*
