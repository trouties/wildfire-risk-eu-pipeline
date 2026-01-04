"""
executive_summary.py — WildfireRisk-EU Output: Executive Memo (Task T12)

Fills the docs/executive_memo.md template with real metrics and outputs
a complete 1-page decision-ready memo to outputs/summaries/executive_memo.md.

Updated for v2 (4-event validation):
  - Model performs reliably on terrain- and wind-driven fires (3 of 4 events)
  - Identifies suburban encroachment as unresolved fire type requiring v3
  - v2 dynamic layer partially compensates for wind-driven events (Mati)
  - 4-event LOEO is more honest than 2-event estimate

Output: outputs/summaries/executive_memo.md
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "wildfire_risk.duckdb"
V2_METRICS_FILE = PROJECT_ROOT / "outputs" / "validation" / "v2_model_metrics.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries"
OUT_FILE = OUT_DIR / "executive_memo.md"

RISK_LABELS = {
    5: "Very High",
    4: "High",
    3: "Medium",
    2: "Low",
    1: "Very Low",
}

# Same classification as validation_report.py
_IN_DIST = {"kalamos_2015", "mati_2018", "varybobi_2021"}
_OOD = {"acharnes_2021"}


def _get_class_distribution() -> dict[int, int]:
    """Return {risk_class: count} for full portfolio."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT risk_class, COUNT(*) FROM risk_scores GROUP BY risk_class ORDER BY risk_class"
    ).fetchall()
    con.close()
    return {int(r[0]): int(r[1]) for r in rows}


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load v2 LOEO metrics (4-event)
    v2 = {}
    if V2_METRICS_FILE.exists():
        with open(V2_METRICS_FILE, encoding="utf-8") as f:
            v2 = json.load(f)

    dist = _get_class_distribution()
    total = sum(dist.values())
    run_date = date.today().isoformat()

    cls_rows = ""
    for c in [5, 4, 3, 2, 1]:
        n = dist.get(c, 0)
        pct = n / total * 100
        notes = {
            5: "Primary attention list — mandatory review",
            4: "Monitor; consider sub-limits at renewal",
            3: "Standard terms",
            2: "Standard terms",
            1: "Preferred risk",
        }[c]
        cls_rows += f"| {c} — {RISK_LABELS[c]} | {n:,} | {pct:.1f}% | {notes} |\n"

    n_cls5 = dist.get(5, 0)
    pct_cls5 = n_cls5 / total * 100

    # Build v2 LOEO validation table
    v2_event_data = {k: v for k, v in v2.items() if k != "_meta"}
    v2_rows = ""
    for name, m in v2_event_data.items():
        display = {
            "mati_2018": "Mati 2018",
            "varybobi_2021": "Varybobi 2021",
            "kalamos_2015": "Kalamos 2015",
            "acharnes_2021": "Acharnes 2021 *",
        }.get(name, name)
        v1_auc = m["auc_structural_v1"]
        v2_auc = m["auc_lgbm_v2"]
        status = "PASS" if v2_auc >= 0.70 else "WARN" if v2_auc >= 0.60 else "FAIL"
        v2_rows += f"| {display} | {v1_auc:.2f} | {v2_auc:.2f} | {status} |\n"

    # In-distribution mean
    id_v1 = [m["auc_structural_v1"] for n, m in v2_event_data.items() if n in _IN_DIST]
    id_v2 = [m["auc_lgbm_v2"] for n, m in v2_event_data.items() if n in _IN_DIST]
    mean_v1 = sum(id_v1) / len(id_v1) if id_v1 else 0
    mean_v2 = sum(id_v2) / len(id_v2) if id_v2 else 0

    memo = f"""# Wildfire Exposure Assessment — Attica Region, Greece

**To**: Portfolio Underwriting Team
**From**: Geospatial Risk Analytics
**Date**: {run_date}
**Re**: Pre-season wildfire risk scores for {total:,} assessed buildings, Attica WUI

---

## Purpose

This memo summarises the WildfireRisk-EU v2 pre-season wildfire exposure assessment
for {total:,} buildings across the Attica Region, Greece.  Scores combine a structural
susceptibility layer (terrain, vegetation, fire weather climatology, fire history) with
an event-context dynamic layer (ERA5-derived wind/VPD/drought + LightGBM).  Scores
support re-pricing, exclusion decisions, and accumulation monitoring ahead of the
June–October fire season.

---

## Portfolio Summary

| Risk Class | Buildings (n) | Portfolio (%) | Recommended Action |
|------------|--------------|---------------|--------------------|
{cls_rows}
**{n_cls5:,} buildings ({pct_cls5:.1f}% of portfolio) are classified Very High Risk.**
These should be the primary focus of pre-season underwriting review.

---

## Validation Summary (4 Events, LOEO Cross-Validation)

The model was backtested against 4 historical wildfire events in Attica using
leave-one-event-out cross-validation (21 features, fire history excluded).

| Event | v1 AUC | v2 AUC | Status |
|-------|--------|--------|--------|
{v2_rows}| **Mean (in-distribution)** | **{mean_v1:.2f}** | **{mean_v2:.2f}** | — |

\\* *Acharnes 2021: suburban encroachment fire — excluded from aggregate metrics
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
| Very High ({n_cls5:,} buildings) | Mandatory pre-renewal site survey; consider wildfire sub-limit or exclusion |
| High ({dist.get(4, 0):,} buildings) | Wildfire premium loading; flag for accumulation monitoring |
| Medium ({dist.get(3, 0):,} buildings) | Standard terms; include in portfolio aggregate CAT model run |
| Low / Very Low | Standard terms |

> **Caveat**: For suburban WUI locations at the wildland edge (Acharnes, Mati coast
> archetype), structural scores may under-rank actual exposure. Apply expert judgment
> for buildings within 500 m of forest/scrubland boundaries in western and eastern Attica.

---

*WildfireRisk-EU v2 | Run date: {run_date} | 4-event LOEO validation*
*Data vintage: ESA WorldCover 2021, EFFIS 2000–2024, ERA5 2015–2024*
*Scores are for risk stratification only and do not constitute actuarial pricing.*
"""

    OUT_FILE.write_text(memo, encoding="utf-8")
    print(f"[executive_summary] Saved memo -> {OUT_FILE}")
    return OUT_FILE


if __name__ == "__main__":
    main()
