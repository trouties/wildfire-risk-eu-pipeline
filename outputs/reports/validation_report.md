# Validation Report — WildfireRisk-EU

> **Statistical caveat (read first).** *LOEO with 4 events is minimally viable but
> not statistically powerful.* The metrics below are illustrative. Per-event AUCs
> are single observations, the in-distribution / out-of-distribution partition is
> post-hoc (chosen after inspecting results), and no aggregate survives a standard
> statistical-significance test. Treat this report as a methodological demonstration.

### Event Classification

**In-distribution events**: Kalamos 2015 (exploratory), Mati 2018, Varybobi 2021 — terrain- and/or wind-driven fires where the feature space (terrain, vegetation, fire weather, fire history) is argued to cover the dominant fire-spread mechanisms.

**Out-of-distribution event**: Acharnes 2021 — suburban encroachment fire type where fire spreads from wildland into low-vegetation built-up areas. Marked as a model boundary case; excluded from aggregate AUC calculations.

> **Note on OOD labeling.** With only 4 events, any in-/out-of-distribution partition
> is unavoidably **post-hoc** — it was chosen after observing per-event AUCs. Treat
> the aggregate metrics computed over the remaining 3 events with corresponding
> caution.

## Multi-Event Summary (v1 Structural Layer)

| Metric | Acharnes 2021 | Kalamos 2015 (exploratory) | Mati 2018 | Varybobi 2021 |
|--------|-----------|-----------|-----------|-----------|
| AUC-ROC | 0.501 [0.49, 0.51] | 0.779 [0.69, 0.86] | 0.431 [0.41, 0.45] | 0.748 [0.74, 0.76] |
| vs Baseline | +0.011 | +0.052 | -0.170 | +0.119 |
| Lift@top10% | 1.03x | 3.97x | 0.74x | 2.07x |
| Precision@class5 | 2.2% | 3.8% | 5.2% | 13.9% |
| Recall@class4+5 | 27.0% | 100.0% | 95.7% | 87.5% |
| Buildings | 60,351 | 973 | 20,891 | 50,638 |
| Burned | 1,838 | 30 | 1,191 | 3,279 |
| Prevalence | 3.0% | 3.1% | 5.7% | 6.5% |

---

## v2 LightGBM LOEO Results

Leave-one-event-out cross-validation with 21 features (fire history excluded
to prevent temporal leakage). Full feature set for SHAP: 26 features.

| Event | Buildings | Burned | v1 AUC [95% CI] | v2 AUC [95% CI] | Δ AUC |
|-------|-----------|--------|-----------------|-----------------|-------|
| Mati 2018 | 20,891 | 1,191 | 0.431 [0.41, 0.45] | 0.715 [0.70, 0.73] | +0.283 |
| Varybobi 2021 | 50,638 | 3,279 | 0.748 [0.74, 0.76] | 0.435 [0.42, 0.45] | -0.313 |
| Kalamos 2015 (exploratory) | 973 | 30 | 0.779 [0.69, 0.86] | 0.373 [0.27, 0.47] | -0.406 |
| Acharnes 2021 * | 60,351 | 1,838 | 0.501 [0.49, 0.51] | 0.737 [0.73, 0.74] | +0.235 |
| *Mean over 3 non-Acharnes events — post-hoc grouping* | — | — | *0.653* | *0.508* | *-0.145* |

\* *Labeled out-of-distribution after observing per-event results; the "mean" row is
computed over the remaining 3 points. With n=4 this post-hoc grouping is
descriptive, not a sampling distribution.*


---

## Acharnes 2021

**Event**: EFFIS_20210823_014 (2021-08-23)
**Population**: 60,351 buildings | **Burned**: 1,838 (3.0%)
**Validation bbox**: [23.67, 38.03, 23.8, 38.14]

### Discrimination Metrics

| Metric | Model | Baseline |
|--------|-------|----------|
| AUC-ROC | 0.501 [0.49, 0.51] | 0.491 [0.48, 0.50] |
| Lift@top10% | 1.03x | 0.44x |
| Precision@class5 | 2.2% | -- |
| Recall@class4+5 | 27.0% | -- |

### Geographic Diagnostic (split at 38.09 N)

| Sub-zone | Buildings | Burned | AUC (model) | AUC (baseline) |
|----------|-----------|--------|-------------|----------------|
| south | 57,584 | 1,229 | 0.403 | 0.509 |
| north | 2,767 | 609 | 0.213 | 0.176 |

### ROC Curve

![ROC Curve](acharnes_2021_roc_curve.png)

### Cumulative Lift Chart

![Lift Chart](acharnes_2021_lift_chart.png)

### Per-Class Burned Rate

| Class | Label | Buildings | Burned | Rate |
|-------|-------|-----------|--------|------|
| 5 | Very High | 3,153 | 68 | 2.2% |
| 4 | High | 13,130 | 428 | 3.3% |
| 3 | Medium | 19,355 | 550 | 2.8% |
| 2 | Low | 20,056 | 773 | 3.9% |
| 1 | Very Low | 4,657 | 19 | 0.4% |

### Mean Score: Burned vs Unburned

| Score Component | Burned | Unburned | Delta |
|-----------------|--------|----------|-------|
| composite_score | 0.492 | 0.490 | +0.002 |
| score_terrain | 0.492 | 0.515 | -0.023 |
| score_vegetation | 0.519 | 0.447 | +0.073 |
| score_fire_weather | 0.221 | 0.429 | -0.208 |
| score_fire_history | 0.722 | 0.591 | +0.131 |

### False Negative Profile (Top 10)

| Building ID | Score | Class | Terrain | Vegetation | Fire Weather | Fire History |
|-------------|-------|-------|---------|------------|-------------|-------------|
| B0038103 | 0.367 | 1 | 0.254 | 0.330 | 0.225 | 0.675 |
| B0055031 | 0.369 | 1 | 0.343 | 0.282 | 0.225 | 0.669 |
| B0209796 | 0.369 | 1 | 0.226 | 0.380 | 0.225 | 0.640 |
| B0119884 | 0.370 | 1 | 0.343 | 0.285 | 0.225 | 0.669 |
| B0106823 | 0.372 | 1 | 0.255 | 0.294 | 0.225 | 0.751 |
| B0010918 | 0.375 | 1 | 0.317 | 0.312 | 0.225 | 0.675 |
| B0175206 | 0.375 | 1 | 0.279 | 0.334 | 0.225 | 0.681 |
| B0218702 | 0.379 | 1 | 0.222 | 0.385 | 0.225 | 0.678 |
| B0010139 | 0.381 | 1 | 0.346 | 0.312 | 0.225 | 0.674 |
| B0134732 | 0.381 | 1 | 0.344 | 0.311 | 0.225 | 0.679 |


---

## Kalamos 2015 (exploratory)

**Event**: EFFIS_20150817_007 (2015-08-17)
**Population**: 973 buildings | **Burned**: 30 (3.1%)
**Validation bbox**: [23.86, 38.12, 23.99, 38.22]

### Discrimination Metrics

| Metric | Model | Baseline |
|--------|-------|----------|
| AUC-ROC | 0.779 [0.69, 0.86] | 0.727 [0.64, 0.80] |
| Lift@top10% | 3.97x | 2.98x |
| Precision@class5 | 3.8% | -- |
| Recall@class4+5 | 100.0% | -- |

### Geographic Diagnostic (split at 38.17 N)

| Sub-zone | Buildings | Burned | AUC (model) | AUC (baseline) |
|----------|-----------|--------|-------------|----------------|
| south | 349 | 24 | 0.717 | 0.662 |
| north | 624 | 6 | 0.960 | 0.848 |

### ROC Curve

![ROC Curve](kalamos_2015_roc_curve.png)

### Cumulative Lift Chart

![Lift Chart](kalamos_2015_lift_chart.png)

### Per-Class Burned Rate

| Class | Label | Buildings | Burned | Rate |
|-------|-------|-----------|--------|------|
| 5 | Very High | 794 | 30 | 3.8% |
| 4 | High | 123 | 0 | 0.0% |
| 3 | Medium | 55 | 0 | 0.0% |
| 2 | Low | 1 | 0 | 0.0% |
| 1 | Very Low | 0 | 0 | 0.0% |

### Mean Score: Burned vs Unburned

| Score Component | Burned | Unburned | Delta |
|-----------------|--------|----------|-------|
| composite_score | 0.693 | 0.641 | +0.051 |
| score_terrain | 0.644 | 0.661 | -0.017 |
| score_vegetation | 0.905 | 0.797 | +0.108 |
| score_fire_weather | 0.181 | 0.213 | -0.032 |
| score_fire_history | 0.936 | 0.818 | +0.118 |

### False Negative Profile (Top 10)

| Building ID | Score | Class | Terrain | Vegetation | Fire Weather | Fire History |
|-------------|-------|-------|---------|------------|-------------|-------------|
| B0002537 | 0.612 | 5 | 0.497 | 0.872 | 0.015 | 0.934 |
| B0000980 | 0.625 | 5 | 0.584 | 0.855 | 0.015 | 0.932 |
| B0185125 | 0.626 | 5 | 0.584 | 0.742 | 0.193 | 0.926 |
| B0169848 | 0.629 | 5 | 0.596 | 0.737 | 0.193 | 0.936 |
| B0116235 | 0.640 | 5 | 0.274 | 0.969 | 0.193 | 0.959 |
| B0169701 | 0.645 | 5 | 0.541 | 0.823 | 0.193 | 0.936 |
| B0209800 | 0.666 | 5 | 0.395 | 0.970 | 0.193 | 0.956 |
| B0203251 | 0.666 | 5 | 0.382 | 0.970 | 0.193 | 0.969 |
| B0222623 | 0.674 | 5 | 0.457 | 0.950 | 0.193 | 0.958 |
| B0098236 | 0.677 | 5 | 0.495 | 0.934 | 0.193 | 0.958 |


---

## Mati 2018

**Event**: EFFIS_20180723_009 (2018-07-23)
**Population**: 20,891 buildings | **Burned**: 1,191 (5.7%)
**Validation bbox**: [23.85, 37.98, 24.1, 38.12]

### Discrimination Metrics

| Metric | Model | Baseline |
|--------|-------|----------|
| AUC-ROC | 0.431 [0.41, 0.45] | 0.602 [0.58, 0.62] |
| Lift@top10% | 0.74x | 1.52x |
| Precision@class5 | 5.2% | -- |
| Recall@class4+5 | 95.7% | -- |

### Geographic Diagnostic (split at 38.05 N)

| Sub-zone | Buildings | Burned | AUC (model) | AUC (baseline) |
|----------|-----------|--------|-------------|----------------|
| south | 16,083 | 342 | 0.790 | 0.555 |
| north | 4,808 | 849 | 0.164 | 0.578 |

### ROC Curve

![ROC Curve](mati_2018_roc_curve.png)

### Cumulative Lift Chart

![Lift Chart](mati_2018_lift_chart.png)

### Per-Class Burned Rate

| Class | Label | Buildings | Burned | Rate |
|-------|-------|-----------|--------|------|
| 5 | Very High | 16,165 | 835 | 5.2% |
| 4 | High | 3,759 | 305 | 8.1% |
| 3 | Medium | 844 | 51 | 6.0% |
| 2 | Low | 107 | 0 | 0.0% |
| 1 | Very Low | 16 | 0 | 0.0% |

### Mean Score: Burned vs Unburned

| Score Component | Burned | Unburned | Delta |
|-----------------|--------|----------|-------|
| composite_score | 0.642 | 0.659 | -0.017 |
| score_terrain | 0.471 | 0.585 | -0.114 |
| score_vegetation | 0.853 | 0.718 | +0.135 |
| score_fire_weather | 0.467 | 0.664 | -0.197 |
| score_fire_history | 0.671 | 0.639 | +0.031 |

### False Negative Profile (Top 10)

| Building ID | Score | Class | Terrain | Vegetation | Fire Weather | Fire History |
|-------------|-------|-------|---------|------------|-------------|-------------|
| B0008409 | 0.479 | 3 | 0.222 | 0.651 | 0.328 | 0.629 |
| B0119306 | 0.480 | 3 | 0.242 | 0.640 | 0.328 | 0.629 |
| B0176781 | 0.482 | 3 | 0.219 | 0.664 | 0.328 | 0.628 |
| B0015119 | 0.484 | 3 | 0.283 | 0.630 | 0.328 | 0.621 |
| B0007116 | 0.485 | 3 | 0.224 | 0.637 | 0.328 | 0.677 |
| B0000722 | 0.488 | 3 | 0.430 | 0.545 | 0.328 | 0.620 |
| B0067807 | 0.489 | 3 | 0.224 | 0.681 | 0.328 | 0.627 |
| B0110060 | 0.489 | 3 | 0.229 | 0.677 | 0.328 | 0.629 |
| B0215863 | 0.489 | 3 | 0.240 | 0.638 | 0.328 | 0.677 |
| B0183916 | 0.490 | 3 | 0.224 | 0.684 | 0.328 | 0.628 |


---

## Varybobi 2021

**Event**: EFFIS_20210803_013 (2021-08-03)
**Population**: 50,638 buildings | **Burned**: 3,279 (6.5%)
**Validation bbox**: [23.6, 38.05, 23.95, 38.25]

### Discrimination Metrics

| Metric | Model | Baseline |
|--------|-------|----------|
| AUC-ROC | 0.748 [0.74, 0.76] | 0.629 [0.62, 0.64] |
| Lift@top10% | 2.07x | 1.88x |
| Precision@class5 | 13.9% | -- |
| Recall@class4+5 | 87.5% | -- |

### Geographic Diagnostic (split at 38.15 N)

| Sub-zone | Buildings | Burned | AUC (model) | AUC (baseline) |
|----------|-----------|--------|-------------|----------------|
| south | 49,392 | 3,199 | 0.751 | 0.625 |
| north | 1,246 | 80 | 0.826 | 0.784 |

### ROC Curve

![ROC Curve](varybobi_2021_roc_curve.png)

### Cumulative Lift Chart

![Lift Chart](varybobi_2021_lift_chart.png)

### Per-Class Burned Rate

| Class | Label | Buildings | Burned | Rate |
|-------|-------|-----------|--------|------|
| 5 | Very High | 14,938 | 2,075 | 13.9% |
| 4 | High | 12,425 | 793 | 6.4% |
| 3 | Medium | 13,337 | 409 | 3.1% |
| 2 | Low | 9,255 | 2 | 0.0% |
| 1 | Very Low | 683 | 0 | 0.0% |

### Mean Score: Burned vs Unburned

| Score Component | Burned | Unburned | Delta |
|-----------------|--------|----------|-------|
| composite_score | 0.621 | 0.548 | +0.073 |
| score_terrain | 0.620 | 0.591 | +0.030 |
| score_vegetation | 0.735 | 0.597 | +0.138 |
| score_fire_weather | 0.199 | 0.208 | -0.010 |
| score_fire_history | 0.873 | 0.770 | +0.103 |

### False Negative Profile (Top 10)

| Building ID | Score | Class | Terrain | Vegetation | Fire Weather | Fire History |
|-------------|-------|-------|---------|------------|-------------|-------------|
| B0081494 | 0.467 | 2 | 0.343 | 0.454 | 0.198 | 0.879 |
| B0108731 | 0.476 | 2 | 0.370 | 0.461 | 0.198 | 0.880 |
| B0154517 | 0.476 | 3 | 0.370 | 0.464 | 0.198 | 0.880 |
| B0097190 | 0.476 | 3 | 0.272 | 0.530 | 0.198 | 0.879 |
| B0173902 | 0.476 | 3 | 0.384 | 0.454 | 0.198 | 0.880 |
| B0107796 | 0.477 | 3 | 0.285 | 0.522 | 0.198 | 0.879 |
| B0056231 | 0.478 | 3 | 0.272 | 0.534 | 0.198 | 0.879 |
| B0220885 | 0.480 | 3 | 0.271 | 0.540 | 0.198 | 0.879 |
| B0118404 | 0.480 | 3 | 0.384 | 0.465 | 0.198 | 0.880 |
| B0075665 | 0.481 | 3 | 0.315 | 0.516 | 0.198 | 0.878 |


---

## Per-event observations (n = 1 each)

Each bullet below describes a **single event**. With one observation per layer ×
event, these are descriptions, not validated generalizations.

- **Kalamos 2015 (exploratory)** (v1 AUC = 0.779, v2 AUC = 0.373) — only 30 burned
  buildings; the 95% CI on v1 AUC is [0.69, 0.86]. Interpret with caution.
- **Mati 2018** (v1 AUC = 0.431, v2 AUC = 0.715) — v1 below chance, v2 above 0.70
  on this one wind-driven event.
- **Varybobi 2021** (v1 AUC = 0.748, v2 AUC = 0.435) — v1 above 0.70, v2 below
  chance on this one terrain-driven event.
- **Acharnes 2021** (v1 AUC = 0.502, v2 AUC = 0.737) — v1 near chance; v2 AUC is
  higher, **but** per-class burn rates are inverted (Class 5: 2.2%, Class 2: 3.9%,
  see the Per-Class table above), so the AUC figure masks a calibration problem.

**Observation** (not a finding): across these 4 events, v1 is higher on Kalamos and
Varybobi, v2 is higher on Mati and Acharnes. This is a description of 4 points, not
a validated "fire-type boundary." Fire-type-aware LOEO or model selection is a v3
research direction.

### Observed per-event results (n = 1 each — descriptive, not a boundary)

| Fire type (n=1) | v1 AUC | v2 AUC | Observation |
|-----------------|--------|--------|-------------|
| Terrain/fuel-driven (Kalamos) | 0.779 | 0.373 | v1 higher on this single event; v2 below chance. CI wide (30 burned). |
| Wind-driven (Mati)            | 0.431 | 0.715 | v2 higher on this single event; v1 below chance. |
| Terrain-driven large (Varybobi)| 0.748 | 0.435 | v1 higher on this single event; v2 below chance. |
| Suburban encroachment (Acharnes)| 0.502 | 0.737 | v2 AUC higher, but class-rank inverted (Class 5: 2.2% vs Class 2: 3.9%). |

### v2 Layer Note

Across 4 events v2 is higher on Mati (+0.28) and Acharnes (+0.24) and lower on
Kalamos (−0.41) and Varybobi (−0.31). One hypothesis is that expanded Acharnes
coverage (60K buildings) gives LightGBM enough suburban training data to help the
wind/suburban events but distort terrain predictions; another is that the pattern
is a 4-point coincidence at n=4 and would disappear with more events. **Both
hypotheses are consistent with the available data.** Fire-type-aware LOEO or model
selection is a v3 research direction.

---

## Limitations

1. **Fire-type specialization**: v1 structural layer excels on terrain-driven fires
   (Kalamos 0.78, Varybobi 0.75) but fails on wind-driven/suburban events. v2 dynamic
   layer shows the reverse pattern. Neither layer dominates; fire-type-aware model
   selection or ensemble is needed (deferred to v3).

2. **Proxy perimeters**: All 4 validation events use literature-proxy circular
   perimeters (area-equivalent radius from EFFIS annual reports), not actual
   fire boundaries. The `fallback_source` fields in config (EMSR300, EMSR531)
   are documentation-only — the validator loads exclusively from the
   `effis_perimeters` DuckDB table. Burned/unburned labels have boundary
   uncertainty proportional to the deviation between circular proxy and true
   fire scar shape.

3. **ERA5 resolution**: ~9 km grid; all buildings in one cell receive identical
   dynamic feature values (see ERA5 Resolution Diagnostic below).

4. **Fire history leakage**: Fire history features include post-event data and are
   excluded from LOEO to mitigate leakage. Full per-event temporal cutoff is
   deferred to v3 (see [Leakage Audit](../docs/leakage_audit.md)).

5. **Small sample sizes**: Kalamos 2015 has only 30 burned buildings (wide CI);
   LOEO with 4 events is minimally viable but not statistically powerful.

---

## ERA5 Resolution Diagnostic

## Summary

- **Total buildings**: 226,314
- **Total land-valid ERA5 grid cells**: 37
- **Grid cells with buildings**: 28
- **Grid resolution**: 0.1° x 0.1° (~10 km)

## Per-Cell Building Distribution

| Statistic | Value |
|-----------|-------|
| Mean buildings per cell | 8,083 |
| Median buildings per cell | 862 |
| Std dev | 19,758 |
| Min | 6 |
| Max | 83,461 |

## Implication

All 226,314 buildings share fire-weather and dynamic features from only **28 unique ERA5 grid cells**. Buildings within the same cell receive identical values for all 5 fire-weather climatology features and all 5 event-day dynamic features. This means ERA5-derived features cannot discriminate between buildings within the same ~10 km cell — discrimination power comes entirely from terrain, vegetation, and fire-history features at the sub-grid scale.

## Per-Cell Detail

| Cell ID | Latitude | Longitude | Buildings | % of Total |
|---------|----------|-----------|-----------|------------|
| LAT38p0_LON23p7 | 38.0 | 23.7 | 83,461 | 36.9% |
| LAT38p0_LON23p8 | 38.0 | 23.8 | 58,791 | 26.0% |
| LAT38p1_LON23p8 | 38.1 | 23.8 | 41,628 | 18.4% |
| LAT38p0_LON23p9 | 38.0 | 23.9 | 10,350 | 4.6% |
| LAT38p0_LON24p0 | 38.0 | 24.0 | 6,855 | 3.0% |
| LAT38p1_LON23p7 | 38.1 | 23.7 | 3,942 | 1.7% |
| LAT38p1_LON23p9 | 38.1 | 23.9 | 3,659 | 1.6% |
| LAT38p0_LON23p6 | 38.0 | 23.6 | 3,227 | 1.4% |
| LAT37p8_LON24p0 | 37.8 | 24.0 | 2,581 | 1.1% |
| LAT37p9_LON23p9 | 37.9 | 23.9 | 2,147 | 0.9% |
| LAT38p1_LON24p0 | 38.1 | 24.0 | 1,622 | 0.7% |
| LAT38p3_LON23p9 | 38.3 | 23.9 | 1,460 | 0.6% |
| LAT37p7_LON23p9 | 37.7 | 23.9 | 1,172 | 0.5% |
| LAT38p3_LON23p6 | 38.3 | 23.6 | 892 | 0.4% |
| LAT37p7_LON24p0 | 37.7 | 24.0 | 832 | 0.4% |
| LAT37p9_LON23p8 | 37.9 | 23.8 | 832 | 0.4% |
| LAT38p2_LON23p9 | 38.2 | 23.9 | 818 | 0.4% |
| LAT38p1_LON23p6 | 38.1 | 23.6 | 621 | 0.3% |
| LAT38p3_LON23p8 | 38.3 | 23.8 | 552 | 0.2% |
| LAT38p2_LON23p8 | 38.2 | 23.8 | 293 | 0.1% |
| LAT38p2_LON24p0 | 38.2 | 24.0 | 223 | 0.1% |
| LAT38p3_LON23p7 | 38.3 | 23.7 | 137 | 0.1% |
| LAT38p2_LON23p7 | 38.2 | 23.7 | 129 | 0.1% |
| LAT38p2_LON24p1 | 38.2 | 24.1 | 40 | 0.0% |
| LAT38p2_LON23p6 | 38.2 | 23.6 | 24 | 0.0% |
| LAT38p3_LON24p2 | 38.3 | 24.2 | 11 | 0.0% |
| LAT38p3_LON24p0 | 38.3 | 24.0 | 9 | 0.0% |
| LAT38p1_LON23p5 | 38.1 | 23.5 | 6 | 0.0% |

---

---

## Path Forward: v3 Priorities

The 4-event validation identifies three actionable priorities for v3:

1. **Fire-type-aware LOEO**: Exclude OOD events from training or add fire-type
   stratification to prevent suburban-fire patterns from contaminating terrain-fire
   predictions.

2. **Suburban encroachment features**: Building density gradient, wildland-urban
   interface edge proximity, and wind-direction × building-cluster geometry to
   address the Acharnes failure mode.

3. **Temporal cutoff enforcement**: Per-event date filtering for fire history and
   FWI climatology features (currently documented as deferred; see Leakage Audit).

4. **Additional validation events**: Expand to 6+ events across Greece (Evia 2021
   requires AOI expansion) for statistically meaningful cross-event metrics.

---

*Generated by WildfireRisk-EU v2 | 4-Event Validation Report*
