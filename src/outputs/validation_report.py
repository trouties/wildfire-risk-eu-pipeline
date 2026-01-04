"""
validation_report.py — WildfireRisk-EU Output: Multi-Event Validation Report

Generates:
  - outputs/reports/{event}_roc_curve.png  — per-event ROC curves
  - outputs/reports/{event}_lift_chart.png — per-event cumulative lift charts
  - outputs/reports/validation_report.md   — combined Markdown report

Reads from:
  - outputs/validation/{event}_metrics.json
  - outputs/validation/{event}_validation_results.csv
  - outputs/validation/{event}_false_negatives.csv
  - outputs/validation/v2_model_metrics.json  (LightGBM LOEO results)

Supports single-event (backward compatible) and multi-event validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = PROJECT_ROOT / "outputs" / "validation"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

RISK_LABELS = {
    "5": "Very High",
    "4": "High",
    "3": "Medium",
    "2": "Low",
    "1": "Very Low",
}

# Display names for events
EVENT_DISPLAY = {
    "mati_2018": "Mati 2018",
    "varybobi_2021": "Varybobi 2021",
    "kalamos_2015": "Kalamos 2015 (exploratory)",
    "acharnes_2021": "Acharnes 2021",
}

# Event classification: in-distribution vs out-of-distribution
IN_DISTRIBUTION_EVENTS = {"kalamos_2015", "mati_2018", "varybobi_2021"}
OOD_EVENTS = {"acharnes_2021"}


# ---------- data loading -------------------------------------------------------

def _load_event_data(event_name: str) -> dict[str, Any]:
    """Load metrics, results, and false negatives for a single event."""
    metrics_file = VAL_DIR / f"{event_name}_metrics.json"
    results_file = VAL_DIR / f"{event_name}_validation_results.csv"
    fn_file = VAL_DIR / f"{event_name}_false_negatives.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_file}")

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(results_file) if results_file.exists() else pd.DataFrame()
    df_fn = pd.read_csv(fn_file) if fn_file.exists() else pd.DataFrame()

    return {"metrics": metrics, "df": df, "df_fn": df_fn}


def _discover_events() -> list[str]:
    """Find all events that have both metrics JSON and validation results CSV."""
    if not VAL_DIR.exists():
        return []
    events = []
    for p in VAL_DIR.glob("*_metrics.json"):
        name = p.stem.replace("_metrics", "")
        # Only include if a matching validation_results CSV exists
        if (VAL_DIR / f"{name}_validation_results.csv").exists():
            events.append(name)
    return sorted(events)


def _load_v2_metrics() -> dict[str, Any] | None:
    """Load v2 LightGBM LOEO metrics if available."""
    v2_path = VAL_DIR / "v2_model_metrics.json"
    if not v2_path.exists():
        return None
    with open(v2_path, encoding="utf-8") as f:
        return json.load(f)


# ---------- plots ---------------------------------------------------------------

def _plot_roc(event_name: str, df: pd.DataFrame, metrics: dict) -> Path:
    """ROC curve: model vs baseline vs random diagonal."""
    burned = df["burned"].astype(int)
    fpr_m, tpr_m, _ = roc_curve(burned, df["composite_score"])
    fpr_b, tpr_b, _ = roc_curve(burned, df["baseline_score"])

    auc_m = metrics["model"]["auc_roc"]
    auc_b = metrics["baseline"]["auc_roc"]
    display = EVENT_DISPLAY.get(event_name, event_name)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_m, tpr_m, lw=2, color="#d73027",
            label=f"WildfireRisk-EU model (AUC = {auc_m:.3f})")
    ax.plot(fpr_b, tpr_b, lw=2, color="#4575b4", linestyle="--",
            label=f"Baseline: dist-to-forest (AUC = {auc_b:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, color="grey", linestyle=":",
            label="Random classifier (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {display} Validation", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path = REPORT_DIR / f"{event_name}_roc_curve.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[validation_report] Saved ROC curve -> {out_path}")
    return out_path


def _plot_lift(event_name: str, df: pd.DataFrame, metrics: dict) -> Path:
    """Cumulative lift chart for model and baseline."""
    burned = df["burned"].astype(int)
    prevalence = burned.mean()
    display = EVENT_DISPLAY.get(event_name, event_name)

    def _lift_series(scores: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        order = scores.argsort()[::-1].values
        burned_sorted = burned.values[order]
        cumulative_rate = np.cumsum(burned_sorted) / (np.arange(len(burned_sorted)) + 1)
        lift = cumulative_rate / prevalence
        pct = np.arange(1, len(burned_sorted) + 1) / len(burned_sorted) * 100
        return pct, lift

    pct_m, lift_m = _lift_series(df["composite_score"])
    pct_b, lift_b = _lift_series(df["baseline_score"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(pct_m, lift_m, lw=2, color="#d73027",
            label=f"Model (lift@10% = {metrics['model']['lift_at_top10pct']:.2f}x)")
    ax.plot(pct_b, lift_b, lw=2, color="#4575b4", linestyle="--",
            label=f"Baseline (lift@10% = {metrics['baseline']['lift_at_top10pct']:.2f}x)")
    ax.axhline(1.0, lw=1, color="grey", linestyle=":", label="Random (lift = 1.0x)")
    ax.set_xlabel("Percentage of buildings scored (top -> bottom)", fontsize=11)
    ax.set_ylabel("Cumulative lift (vs random)", fontsize=11)
    ax.set_title(f"Cumulative Lift Chart — {display} Validation", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    out_path = REPORT_DIR / f"{event_name}_lift_chart.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[validation_report] Saved lift chart -> {out_path}")
    return out_path


# ---------- report sections ----------------------------------------------------

def _section_summary_matrix(events: dict[str, dict]) -> str:
    """Generate the multi-event comparison matrix."""
    header = "| Metric |"
    sep = "|--------|"
    rows = {
        "AUC-ROC": "| AUC-ROC |",
        "vs Baseline": "| vs Baseline |",
        "Lift@10%": "| Lift@top10% |",
        "Precision@cls5": "| Precision@class5 |",
        "Recall@cls4+5": "| Recall@class4+5 |",
        "Buildings": "| Buildings |",
        "Burned": "| Burned |",
        "Prevalence": "| Prevalence |",
    }

    for name in events:
        m = events[name]["metrics"]
        display = EVENT_DISPLAY.get(name, name)
        header += f" {display} |"
        sep += "-----------|"

        auc = m["model"]["auc_roc"]
        auc_ci_lo = m["model"].get("auc_roc_ci_lower")
        auc_ci_hi = m["model"].get("auc_roc_ci_upper")
        status = "PASS" if auc >= 0.70 else "WARN" if auc >= 0.60 else "FAIL"
        delta = m["model_vs_baseline"]["auc_roc_delta"]
        delta_sign = "+" if delta > 0 else ""

        ci_str = f" [{auc_ci_lo:.2f}, {auc_ci_hi:.2f}]" if auc_ci_lo is not None else ""
        rows["AUC-ROC"] += f" **{auc:.3f}**{ci_str} ({status}) |"
        rows["vs Baseline"] += f" {delta_sign}{delta:.3f} |"
        rows["Lift@10%"] += f" {m['model']['lift_at_top10pct']:.2f}x |"
        rows["Precision@cls5"] += f" {m['model']['precision_cls5']:.1%} |"
        rows["Recall@cls4+5"] += f" {m['model']['recall_cls45']:.1%} |"
        rows["Buildings"] += f" {m['population']['n_buildings']:,} |"
        rows["Burned"] += f" {m['population']['n_burned']:,} |"
        rows["Prevalence"] += f" {m['population']['prevalence']:.1%} |"

    lines = [header, sep] + list(rows.values())
    return "\n".join(lines)


def _section_event_detail(event_name: str, data: dict) -> str:
    """Generate a per-event detail section."""
    m = data["metrics"]
    df_fn = data["df_fn"]
    display = EVENT_DISPLAY.get(event_name, event_name)

    auc_m = m["model"]["auc_roc"]
    auc_b = m["baseline"]["auc_roc"]
    lift_m = m["model"]["lift_at_top10pct"]
    lift_b = m["baseline"]["lift_at_top10pct"]
    pop = m["population"]
    cls = m["class_stats"]
    geo = m.get("geographic_diagnostic", {})

    # CI strings
    auc_m_ci_lo = m["model"].get("auc_roc_ci_lower")
    auc_m_ci_hi = m["model"].get("auc_roc_ci_upper")
    auc_b_ci_lo = m["baseline"].get("auc_roc_ci_lower")
    auc_b_ci_hi = m["baseline"].get("auc_roc_ci_upper")
    m_ci_str = f" [{auc_m_ci_lo:.2f}, {auc_m_ci_hi:.2f}]" if auc_m_ci_lo is not None else ""
    b_ci_str = f" [{auc_b_ci_lo:.2f}, {auc_b_ci_hi:.2f}]" if auc_b_ci_lo is not None else ""

    # Status
    status = "PASS" if auc_m >= 0.70 else "WARN" if auc_m >= 0.60 else "FAIL"

    # Class table
    class_rows = ""
    for c in ["5", "4", "3", "2", "1"]:
        s = cls[c]
        class_rows += (
            f"| {c} | {RISK_LABELS[c]} | {s['n_buildings']:,} | "
            f"{s['n_burned']:,} | {s['burned_rate']:.1%} |\n"
        )

    # Geographic diagnostic
    geo_section = ""
    if geo:
        geo_rows = ""
        for zone, g in geo.items():
            zone_auc = g.get("auc_roc_model", float("nan"))
            base_auc = g.get("auc_roc_baseline", float("nan"))
            zone_status = "PASS" if zone_auc >= 0.70 else "WARN" if zone_auc >= 0.60 else "FAIL"
            geo_rows += (
                f"| {zone} | {g['n_buildings']:,} | {g['n_burned']:,} | "
                f"**{zone_auc:.3f}** | {base_auc:.3f} | {zone_status} |\n"
            )
        split_lat = next(
            (v.get("lat_threshold") for v in geo.values() if "lat_threshold" in v),
            None,
        )
        split_note = f" (split at {split_lat} N)" if split_lat else ""
        geo_section = f"""
### Geographic Diagnostic{split_note}

| Sub-zone | Buildings | Burned | AUC (model) | AUC (baseline) | Status |
|----------|-----------|--------|-------------|----------------|--------|
{geo_rows}"""

    # False negatives
    fn_rows = ""
    if not df_fn.empty:
        for _, row in df_fn.head(10).iterrows():
            fn_rows += (
                f"| {row['building_id']} | {row['composite_score']:.3f} | "
                f"{int(row['risk_class'])} | {row['score_terrain']:.3f} | "
                f"{row['score_vegetation']:.3f} | {row['score_fire_weather']:.3f} | "
                f"{row['score_fire_history']:.3f} |\n"
            )

    # Score differences
    ms_b = m["mean_scores"]["burned"]
    ms_u = m["mean_scores"]["unburned"]
    score_rows = ""
    for col in ["composite_score", "score_terrain", "score_vegetation",
                "score_fire_weather", "score_fire_history"]:
        diff = ms_b[col] - ms_u[col]
        score_rows += (
            f"| {col} | {ms_b[col]:.3f} | {ms_u[col]:.3f} | {diff:+.3f} |\n"
        )

    return f"""## {display}

**Event**: {m['event']} ({m['event_date']})
**Population**: {pop['n_buildings']:,} buildings | **Burned**: {pop['n_burned']:,} ({pop['prevalence']:.1%})
**Validation bbox**: [{', '.join(str(v) for v in m['validation_bbox'])}]

### Discrimination Metrics

| Metric | Model | Baseline | Status |
|--------|-------|----------|--------|
| AUC-ROC | **{auc_m:.3f}**{m_ci_str} | {auc_b:.3f}{b_ci_str} | {status} |
| Lift@top10% | {lift_m:.2f}x | {lift_b:.2f}x | {'BEAT' if lift_m > lift_b else 'BELOW'} |
| Precision@class5 | {m['model']['precision_cls5']:.1%} | -- | -- |
| Recall@class4+5 | {m['model']['recall_cls45']:.1%} | -- | -- |
{geo_section}
### ROC Curve

![ROC Curve]({event_name}_roc_curve.png)

### Cumulative Lift Chart

![Lift Chart]({event_name}_lift_chart.png)

### Per-Class Burned Rate

| Class | Label | Buildings | Burned | Rate |
|-------|-------|-----------|--------|------|
{class_rows}
### Mean Score: Burned vs Unburned

| Score Component | Burned | Unburned | Delta |
|-----------------|--------|----------|-------|
{score_rows}
### False Negative Profile (Top 10)

| Building ID | Score | Class | Terrain | Vegetation | Fire Weather | Fire History |
|-------------|-------|-------|---------|------------|-------------|-------------|
{fn_rows}"""


def _section_event_classification(events: dict[str, dict]) -> str:
    """Generate the event classification preamble above the summary matrix."""
    if len(events) < 2:
        return ""

    id_names = [n for n in events if n in IN_DISTRIBUTION_EVENTS]
    ood_names = [n for n in events if n in OOD_EVENTS]

    id_bullets = ", ".join(EVENT_DISPLAY.get(n, n) for n in id_names)
    ood_bullets = ", ".join(EVENT_DISPLAY.get(n, n) for n in ood_names)

    lines = ["### Event Classification\n"]

    if id_names:
        lines.append(
            f"**In-distribution events**: {id_bullets} — terrain- and/or wind-driven "
            "fires where the feature space (terrain, vegetation, fire weather, fire "
            "history) covers the dominant fire-spread mechanisms.\n"
        )
    if ood_names:
        lines.append(
            f"**Out-of-distribution event**: {ood_bullets} — suburban encroachment "
            "fire type where fire spreads from wildland into low-vegetation built-up "
            "areas. Marked as a **model boundary case**; excluded from aggregate AUC "
            "calculations.\n"
        )
    return "\n".join(lines)


def _section_v2_loeo(v2_metrics: dict[str, Any] | None) -> str:
    """Generate the v2 LightGBM LOEO results table."""
    if v2_metrics is None:
        return ""

    meta = v2_metrics.get("_meta", {})
    event_metrics = {k: v for k, v in v2_metrics.items() if k != "_meta"}
    if not event_metrics:
        return ""

    rows = ""
    for name, m in event_metrics.items():
        display = EVENT_DISPLAY.get(name, name)
        v1 = m["auc_structural_v1"]
        v1_ci = f"[{m['auc_structural_v1_ci_lower']:.2f}, {m['auc_structural_v1_ci_upper']:.2f}]"
        v2 = m["auc_lgbm_v2"]
        v2_ci = f"[{m['auc_lgbm_v2_ci_lower']:.2f}, {m['auc_lgbm_v2_ci_upper']:.2f}]"
        delta = m["auc_delta"]
        sign = "+" if delta >= 0 else ""

        annotation = ""
        if name in OOD_EVENTS:
            annotation = " *"

        rows += (
            f"| {display}{annotation} | {m['n_buildings']:,} | {m['n_burned']:,} | "
            f"{v1:.3f} {v1_ci} | {v2:.3f} {v2_ci} | {sign}{delta:.3f} |\n"
        )

    # Compute in-distribution aggregate
    id_v1_aucs = [m["auc_structural_v1"] for n, m in event_metrics.items()
                  if n in IN_DISTRIBUTION_EVENTS]
    id_v2_aucs = [m["auc_lgbm_v2"] for n, m in event_metrics.items()
                  if n in IN_DISTRIBUTION_EVENTS]
    if id_v1_aucs:
        mean_v1 = sum(id_v1_aucs) / len(id_v1_aucs)
        mean_v2 = sum(id_v2_aucs) / len(id_v2_aucs)
        mean_delta = mean_v2 - mean_v1
        sign = "+" if mean_delta >= 0 else ""
        rows += (
            f"| **Mean (in-distribution)** | — | — | "
            f"**{mean_v1:.3f}** | **{mean_v2:.3f}** | **{sign}{mean_delta:.3f}** |\n"
        )

    n_loeo = meta.get("loeo_n_features", "?")
    n_shap = meta.get("shap_n_features", "?")

    return f"""## v2 LightGBM LOEO Results

Leave-one-event-out cross-validation with {n_loeo} features (fire history excluded
to prevent temporal leakage). Full feature set for SHAP: {n_shap} features.

| Event | Buildings | Burned | v1 AUC [95% CI] | v2 AUC [95% CI] | Δ AUC |
|-------|-----------|--------|-----------------|-----------------|-------|
{rows}
\\* *Excluded from aggregate metrics; suburban encroachment fire type not represented
in feature space (see Limitations).*

"""


def _section_interpretation(events: dict[str, dict],
                            v2_metrics: dict[str, Any] | None) -> str:
    """Interpretation section with cross-event analysis and v2 degradation explanation."""
    if len(events) < 2:
        return ""

    event_names = list(events.keys())
    aucs = {n: events[n]["metrics"]["model"]["auc_roc"] for n in event_names}

    # Identify best/worst among in-distribution events
    id_aucs = {n: aucs[n] for n in event_names if n in IN_DISTRIBUTION_EVENTS}
    if id_aucs:
        best_event = max(id_aucs, key=id_aucs.get)
        worst_id_event = min(id_aucs, key=id_aucs.get)
    else:
        best_event = max(aucs, key=aucs.get)
        worst_id_event = min(aucs, key=aucs.get)

    best_display = EVENT_DISPLAY.get(best_event, best_event)
    worst_id_display = EVENT_DISPLAY.get(worst_id_event, worst_id_event)

    ood_section = ""
    for name in event_names:
        if name in OOD_EVENTS:
            ood_display = EVENT_DISPLAY.get(name, name)
            ood_auc = aucs[name]
            ood_section += f"""
- **{ood_display}** (AUC = {ood_auc:.3f}): Out-of-distribution. The model is inverted
  (AUC < 0.50) — it assigns *lower* risk scores to buildings that burned. The fire
  spread from wildland into a low-vegetation suburban zone; the model's vegetation
  and fire-history features point away from the actual burn area. This fire type
  (suburban encroachment) is outside the model's design envelope.
"""

    # v2 degradation explanation for Varybobi
    v2_note = ""
    if v2_metrics and "varybobi_2021" in v2_metrics and "mati_2018" in v2_metrics:
        v2_varybobi = v2_metrics["varybobi_2021"]["auc_lgbm_v2"]
        v2_note = f"""
### v2 LOEO Degradation Note

The 2-event LOEO v2 AUC of 0.746 (Mati) and 0.712 (Varybobi) overestimated
generalization. With 4 events, Varybobi v2 AUC degrades to {v2_varybobi:.3f}
due to out-of-distribution training contamination from Acharnes 2021 — the
LightGBM model learns suburban-fire patterns from Acharnes that actively
mislead predictions on Varybobi's terrain-driven fire. The 4-event result
is more conservative and honest; it reveals that the v2 dynamic layer does not
yet generalize across heterogeneous fire types without fire-type-aware training.
"""

    return f"""## Interpretation

The structural susceptibility model shows **differential performance across fire types**:

- **{best_display}** (AUC = {aucs[best_event]:.3f}): The model discriminates effectively
  on this event, where fire spread was primarily driven by terrain and fuel structure —
  exactly the features the structural layer captures.

- **{worst_id_display}** (AUC = {aucs[worst_id_event]:.3f}): The model struggles on
  wind-driven events where acute meteorological conditions override structural risk factors.
{ood_section}
### Applicability Boundaries

| Fire type | Model performance | Root cause |
|-----------|-------------------|------------|
| Terrain/fuel-driven (Kalamos) | PASS (AUC > 0.70) | Vegetation + fire history features align with fire spread |
| Wind-driven (Mati) | FAIL (v1) / WARN (v2) | Wind-driven ember transport overrides structural factors; v2 dynamic layer partially compensates |
| Terrain-driven large (Varybobi) | WARN (v1) / FAIL (v2 4-event) | v1 captures structural signal; v2 LOEO contaminated by OOD training data |
| Suburban encroachment (Acharnes) | FAIL (both layers) | Fire spreads into low-vegetation built-up area; outside model design envelope |
{v2_note}"""


def _section_limitations() -> str:
    """Limitations section."""
    return """## Limitations

1. **Model boundary — suburban encroachment fires**: Fires that spread from wildland
   into low-vegetation suburban areas (e.g. Acharnes 2021, v1 AUC = 0.24) are not
   discriminated by current terrain/vegetation/weather features. Building density
   gradients and wind-geometry interaction features are required for this fire type
   (deferred to v3).

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
"""


def _section_resolution_diagnostic() -> str:
    """Include ERA5 resolution diagnostic if the file exists."""
    diag_md = VAL_DIR / "era5_resolution_diagnostic.md"
    if not diag_md.exists():
        return ""

    content = diag_md.read_text(encoding="utf-8")
    # Strip the top-level heading (already in the report) and the trailing generator line
    lines = content.split("\n")
    # Remove the first "# ERA5 Resolution Diagnostic" heading, replace with ##
    body_lines = []
    for line in lines:
        if line.startswith("# ERA5 Resolution Diagnostic"):
            body_lines.append("## ERA5 Resolution Diagnostic")
        elif line.startswith("*Generated by"):
            continue
        else:
            body_lines.append(line)

    return "\n".join(body_lines) + "\n"


def _section_path_forward() -> str:
    """Path forward to v3."""
    return """## Path Forward: v3 Priorities

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
"""


# ---------- main ----------------------------------------------------------------

def _write_report(events: dict[str, dict],
                   v2_metrics: dict[str, Any] | None = None) -> None:
    """Write combined multi-event validation report."""
    sections = [
        "# Validation Report — WildfireRisk-EU\n",
    ]

    # Event classification preamble
    event_class = _section_event_classification(events)
    if event_class:
        sections.append(event_class)

    # Summary matrix (v1 structural)
    sections.append("## Multi-Event Summary (v1 Structural Layer)\n")
    sections.append(_section_summary_matrix(events))
    sections.append("\n---\n")

    # v2 LOEO table
    v2_loeo = _section_v2_loeo(v2_metrics)
    if v2_loeo:
        sections.append(v2_loeo)
        sections.append("---\n")

    # Per-event detail sections
    for name, data in events.items():
        sections.append(_section_event_detail(name, data))
        sections.append("\n---\n")

    # Interpretation (replaces old cross-event analysis)
    interpretation = _section_interpretation(events, v2_metrics)
    if interpretation:
        sections.append(interpretation)
        sections.append("---\n")

    # Limitations
    sections.append(_section_limitations())
    sections.append("---\n")

    # Resolution diagnostic (if generated)
    res_diag = _section_resolution_diagnostic()
    if res_diag:
        sections.append(res_diag)
        sections.append("---\n")

    # Path forward
    sections.append(_section_path_forward())

    report = "\n".join(sections)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[validation_report] Saved report -> {report_path}")


def main(event_names: list[str] | None = None) -> Path:
    """Generate multi-event validation report.

    Args:
        event_names: List of event keys (e.g. ["mati_2018", "varybobi_2021"]).
            If None, auto-discovers all events with metrics in outputs/validation/.

    Returns:
        Path to the generated report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if event_names is None:
        event_names = _discover_events()
    if not event_names:
        raise FileNotFoundError("No validation metrics found in outputs/validation/")

    print(f"[validation_report] Generating report for {len(event_names)} event(s): "
          f"{', '.join(event_names)}")

    # Load data and generate per-event plots
    events: dict[str, dict] = {}
    for name in event_names:
        data = _load_event_data(name)
        if not data["df"].empty:
            _plot_roc(name, data["df"], data["metrics"])
            _plot_lift(name, data["df"], data["metrics"])
        events[name] = data

    # Load v2 LOEO metrics if available
    v2_metrics = _load_v2_metrics()

    # Write combined report
    _write_report(events, v2_metrics=v2_metrics)

    report_path = REPORT_DIR / "validation_report.md"
    return report_path


if __name__ == "__main__":
    import sys
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)
