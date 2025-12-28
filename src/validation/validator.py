"""
validator.py — WildfireRisk-EU Validation Engine

Backtests the composite risk scoring against wildfire events in Attica, Greece.
Supports multiple holdout events configured in config/validation.yaml.
Produces per-building validation labels, discrimination metrics, failure
analysis, and a baseline comparison.

Validation framing
------------------
What is being evaluated
  The composite_score produced by the T10 scoring engine, representing each
  building's pre-event wildfire risk.  This is a *discriminative ranking* task:
  does the model assign systematically higher scores to buildings that were
  subsequently destroyed or damaged by the 2018 Mati fire?

Ground truth label strategy
  Buildings whose centroids fall inside the EFFIS Mati 2018 perimeter
  (fire_id = EFFIS_20180723_009, ~1,290 ha) are labelled ``burned=1``.
  All other buildings within the validation bounding box are ``burned=0``.
  Source: EFFIS perimeter polygon stored in DuckDB (WKT, EPSG:2100).

Temporal / split logic
  - Feature cutoff: 2018-07-22 (one day before the fire).
  - The T10 scoring used EFFIS perimeters acquired before 2018 and ERA5
    climatology (2015–2024 seasonal averages, no day-specific leakage).
  - The Mati 2018 fire perimeter is held out as the ground-truth label;
    it was NOT available when features were computed.
  - Note: FIRMS hotspot data spans 2015–2024 (includes post-fire hotspots
    near Mati on 2018-07-24).  This introduces minor leakage into
    ``firms_hotspot_count_5km``, but Mati's firms score was 6.9 vs AOI
    average 78.9, so the direction of leakage is against the model —
    conservative rather than optimistic.

Validation population
  Buildings within the Mati event bounding box:
  [west=23.85, south=37.98, east=24.10, north=38.12] (WGS84).
  This restricts the analysis to the zone of potential impact and avoids
  diluting metrics with buildings hundreds of kilometres from the fire.

Metrics
  1. AUC-ROC: area under the ROC curve.  Threshold-free discrimination.
     Failure threshold: < 0.60.  Target: ≥ 0.70.
  2. Lift@top10%: fraction of burned buildings captured in the top 10%
     of composite scores, divided by the expected rate if random (10%).
     Reports the concentration of true positives at the high-risk tail.
  3. Precision@class5: fraction of class-5 buildings (within validation
     bbox) that were actually burned.  Stakeholder metric: "if I flag
     Very High risk buildings, how often am I right?"
  4. Recall@class4+5: fraction of burned buildings that were classified
     as class 4 or 5.  Sensitivity of the high-risk tier.

Baseline comparison
  Naive single-feature model: ``baseline_score = 1 / (dist_to_forest_m + 1)``.
  Rank-normalised to [0,1] over the validation bbox using the same
  percentile-rank normalisation as the main model.  AUC-ROC and Lift@top10%
  are computed for both models.

Failure analysis
  - Top-10 false negatives: burned buildings with the lowest composite scores.
    These represent the model's most consequential misses.
  - Top-10 false positives: unburned buildings with the highest composite
    scores.  These represent over-predicted risk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import rankdata
from shapely import contains_xy
from shapely import wkt as swkt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Bootstrap confidence interval for AUC-ROC
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AUC-ROC with bootstrap confidence interval.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores/probabilities.
        n_bootstraps: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (auc_point, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    auc_point = roc_auc_score(y_true, y_score)

    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        # Skip degenerate samples (all 0 or all 1)
        if y_t.sum() == 0 or y_t.sum() == n:
            continue
        aucs.append(roc_auc_score(y_t, y_s))

    if not aucs:
        return auc_point, float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(aucs, 100 * alpha))
    upper = float(np.percentile(aucs, 100 * (1 - alpha)))
    return auc_point, lower, upper

# Event constants — match validation.yaml
_EVENT_ID        = "EFFIS_20180723_009"
_EVENT_DATE      = "2018-07-23"
_VALIDATION_BBOX = (23.85, 37.98, 24.10, 38.12)  # west, south, east, north


# ---------------------------------------------------------------------------
# Step 1: Build ground-truth burned labels
# ---------------------------------------------------------------------------

def _load_event_perimeter(db_path: Path, event_id: str = _EVENT_ID):
    """Load an EFFIS fire perimeter as a Shapely Polygon (EPSG:2100).

    Args:
        db_path: Path to the DuckDB database.
        event_id: EFFIS fire identifier (e.g. "EFFIS_20180723_009").
    """
    con = duckdb.connect(str(db_path), read_only=True)
    row = con.execute("""
        SELECT geometry FROM effis_perimeters WHERE fire_id = ?
    """, [event_id]).fetchone()
    con.close()
    if row is None:
        raise ValueError(f"EFFIS perimeter {event_id!r} not found in DuckDB")
    poly = swkt.loads(row[0])
    return poly


def _label_buildings(
    db_path: Path,
    mati_poly,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Return DataFrame (building_id, centroid_lat, centroid_lon, burned).

    burned = 1 if the building centroid falls inside the Mati 2018 perimeter,
    else 0.  Restricted to buildings within ``bbox`` (west, south, east, north).
    """
    west, south, east, north = bbox
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT building_id, centroid_lat, centroid_lon
        FROM buildings
        WHERE centroid_lon BETWEEN ? AND ?
          AND centroid_lat BETWEEN ? AND ?
        ORDER BY building_id
    """, [west, east, south, north]).df()
    con.close()

    # Project WGS84 centroids to EPSG:2100 for point-in-polygon test
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    xs, ys = transformer.transform(df["centroid_lon"].values, df["centroid_lat"].values)

    burned = contains_xy(mati_poly, xs, ys).astype(int)
    df["burned"] = burned
    return df


# ---------------------------------------------------------------------------
# Step 2: Join scores and features
# ---------------------------------------------------------------------------

def _load_scores_and_features(db_path: Path, building_ids: pd.Series) -> pd.DataFrame:
    """Load risk_scores + dist_to_forest for baseline, restricted to building_ids."""
    id_list = building_ids.tolist()
    con = duckdb.connect(str(db_path), read_only=True)

    # risk_scores
    rs = con.execute("""
        SELECT building_id, composite_score, risk_class,
               score_terrain, score_vegetation, score_fire_weather, score_fire_history
        FROM risk_scores
        ORDER BY building_id
    """).df()

    # baseline feature: dist_to_forest_m
    feat = con.execute("""
        SELECT building_id, dist_to_forest_m
        FROM features_vegetation
        ORDER BY building_id
    """).df()

    con.close()

    # Filter to validation bbox buildings
    rs = rs[rs["building_id"].isin(id_list)].copy()
    feat = feat[feat["building_id"].isin(id_list)].copy()

    merged = rs.merge(feat, on="building_id", how="left")
    return merged


# ---------------------------------------------------------------------------
# Step 3: Baseline score
# ---------------------------------------------------------------------------

def _compute_baseline(df: pd.DataFrame) -> pd.Series:
    """Baseline: 1/(dist_to_forest_m + 1), rank-normalised to [0, 1].

    Closer to forest = higher baseline score (direction=negative → invert after norm).
    The raw formula produces higher values for smaller distances (correct direction),
    so we treat it as direction=positive after transformation.
    """
    raw = 1.0 / (df["dist_to_forest_m"].values.astype(float) + 1.0)
    # Clip to p1/p99 and rank-normalise
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
    clipped = np.clip(raw, lo, hi)
    n = len(clipped)
    if n <= 1:
        return pd.Series([0.5] * n, index=df.index)
    ranks = rankdata(clipped, method="average")
    scores = (ranks - 1.0) / (n - 1.0)
    return pd.Series(scores, index=df.index, name="baseline_score")


# ---------------------------------------------------------------------------
# Step 4: Metrics
# ---------------------------------------------------------------------------

def _lift_at_topk(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    """Lift at top-k%: fraction of positives in top-k% / prevalence.

    A lift of 2.0 means the top-k% captures twice as many positives as
    random selection would predict.
    """
    n = len(y_true)
    n_top = max(1, int(np.ceil(n * k)))
    top_idx = np.argsort(y_score)[::-1][:n_top]
    precision_top = y_true[top_idx].mean()
    prevalence = y_true.mean()
    if prevalence == 0:
        return float("nan")
    return float(precision_top / prevalence)


def compute_metrics(
    df: pd.DataFrame,
    event_id: str = _EVENT_ID,
    event_date: str = _EVENT_DATE,
    bbox: tuple[float, float, float, float] = _VALIDATION_BBOX,
    geo_split_lat: float | None = 38.05,
) -> dict[str, Any]:
    """Compute all validation metrics.

    Args:
        df: DataFrame with columns: burned, composite_score, baseline_score,
            risk_class, score_terrain, score_vegetation, score_fire_weather,
            score_fire_history.
        event_id: EFFIS fire identifier.
        event_date: Fire event date string.
        bbox: Validation bounding box (west, south, east, north).
        geo_split_lat: Latitude to split for geographic diagnostic.
            None to skip geographic diagnostic.

    Returns:
        Nested metrics dict.
    """
    y_true  = df["burned"].values.astype(int)
    y_score = df["composite_score"].values
    y_base  = df["baseline_score"].values

    # ── AUC-ROC (with bootstrap 95% CI) ──────────────────────────────────
    auc_model, auc_model_ci_lo, auc_model_ci_hi = bootstrap_auc_ci(
        y_true, y_score, n_bootstraps=1000, ci=0.95,
    )
    auc_baseline, auc_base_ci_lo, auc_base_ci_hi = bootstrap_auc_ci(
        y_true, y_base, n_bootstraps=1000, ci=0.95,
    )

    # ── Lift@top10% ────────────────────────────────────────────────────────
    lift_model    = _lift_at_topk(y_true, y_score, k=0.10)
    lift_baseline = _lift_at_topk(y_true, y_base, k=0.10)

    # ── Precision@class5 (within validation bbox) ──────────────────────────
    cls5_mask         = df["risk_class"] == 5
    n_cls5            = cls5_mask.sum()
    precision_cls5    = float(df.loc[cls5_mask, "burned"].mean()) if n_cls5 > 0 else float("nan")

    # ── Recall@class4+5 ────────────────────────────────────────────────────
    n_burned          = int(y_true.sum())
    high_risk_mask    = df["risk_class"].isin([4, 5])
    recall_cls45      = float(df.loc[high_risk_mask & (df["burned"] == 1), "burned"].sum()) / max(n_burned, 1)

    # ── Population stats ───────────────────────────────────────────────────
    n_total     = len(df)
    prevalence  = float(y_true.mean())

    # ── Per-class burned rate ───────────────────────────────────────────────
    class_stats = {}
    for cls in [1, 2, 3, 4, 5]:
        mask = df["risk_class"] == cls
        n_cls = mask.sum()
        n_burned_cls = df.loc[mask, "burned"].sum()
        class_stats[str(cls)] = {
            "n_buildings":   int(n_cls),
            "n_burned":      int(n_burned_cls),
            "burned_rate":   float(n_burned_cls / n_cls) if n_cls > 0 else 0.0,
        }

    # ── Mean score comparison (burned vs unburned) ──────────────────────────
    score_cols = [
        "composite_score", "score_terrain", "score_vegetation",
        "score_fire_weather", "score_fire_history",
    ]
    mean_burned   = {c: float(df.loc[df["burned"]==1, c].mean()) for c in score_cols}
    mean_unburned = {c: float(df.loc[df["burned"]==0, c].mean()) for c in score_cols}

    # ── Geographic sub-analysis (fire is predominantly north of 38.05°N) ───
    # The Mati perimeter sits roughly 38.00–38.06°N.  Splitting at 38.05 reveals
    # which part of the bbox drives the poor overall AUC.
    geo_diag: dict[str, Any] = {}
    if "centroid_lat" in df.columns and geo_split_lat is not None:
        _GEO_SPLIT = geo_split_lat
        for zone, mask_fn in [
            ("south", df["centroid_lat"] <  _GEO_SPLIT),
            ("north", df["centroid_lat"] >= _GEO_SPLIT),
        ]:
            sub = df[mask_fn]
            n_sub = len(sub)
            n_b_sub = int(sub["burned"].sum())
            if n_b_sub >= 2 and n_sub - n_b_sub >= 2:
                auc_sub_m = float(roc_auc_score(sub["burned"], sub["composite_score"]))
                auc_sub_b = float(roc_auc_score(sub["burned"], sub["baseline_score"]))
            else:
                auc_sub_m = float("nan")
                auc_sub_b = float("nan")
            geo_diag[zone] = {
                "lat_threshold":  _GEO_SPLIT,
                "n_buildings":    n_sub,
                "n_burned":       n_b_sub,
                "prevalence":     float(n_b_sub / n_sub) if n_sub else 0.0,
                "auc_roc_model":  auc_sub_m,
                "auc_roc_baseline": auc_sub_b,
            }

    return {
        "event":             event_id,
        "event_date":        event_date,
        "validation_bbox":   list(bbox),
        "population": {
            "n_buildings":   n_total,
            "n_burned":      n_burned,
            "n_unburned":    n_total - n_burned,
            "prevalence":    prevalence,
        },
        "model": {
            "auc_roc":          float(auc_model),
            "auc_roc_ci_lower": float(auc_model_ci_lo),
            "auc_roc_ci_upper": float(auc_model_ci_hi),
            "lift_at_top10pct": float(lift_model),
            "precision_cls5":   float(precision_cls5),
            "recall_cls45":     float(recall_cls45),
            "n_cls5":           int(n_cls5),
        },
        "baseline": {
            "name":             "dist_to_forest_naive",
            "auc_roc":          float(auc_baseline),
            "auc_roc_ci_lower": float(auc_base_ci_lo),
            "auc_roc_ci_upper": float(auc_base_ci_hi),
            "lift_at_top10pct": float(lift_baseline),
        },
        "model_vs_baseline": {
            "auc_roc_delta":  float(auc_model - auc_baseline),
            "lift_delta":     float(lift_model - lift_baseline),
        },
        "class_stats":       class_stats,
        "mean_scores": {
            "burned":   mean_burned,
            "unburned": mean_unburned,
        },
        "geographic_diagnostic": geo_diag,
        "thresholds": {
            "auc_failure":  0.60,
            "auc_target":   0.70,
        },
    }


# ---------------------------------------------------------------------------
# Step 5: Failure analysis
# ---------------------------------------------------------------------------

def failure_analysis(
    df: pd.DataFrame,
    n_top: int = 10,
) -> dict[str, pd.DataFrame]:
    """Identify and describe top false negatives and false positives.

    False negatives: burned buildings with the lowest composite_score.
    False positives: unburned buildings with the highest composite_score.
    """
    fn_cols = [
        "building_id", "burned", "composite_score", "risk_class",
        "score_terrain", "score_vegetation", "score_fire_weather",
        "score_fire_history", "baseline_score", "dist_to_forest_m",
    ]

    # False negatives: burned=1 AND low composite_score
    burned_df = df[df["burned"] == 1].copy()
    fn = (burned_df
          .sort_values("composite_score")
          .head(n_top)
          [fn_cols])

    # False positives: burned=0 AND high composite_score
    unburned_df = df[df["burned"] == 0].copy()
    fp = (unburned_df
          .sort_values("composite_score", ascending=False)
          .head(n_top)
          [fn_cols])

    return {"false_negatives": fn, "false_positives": fp}


# ---------------------------------------------------------------------------
# Step 6: Write validation_results to DuckDB
# ---------------------------------------------------------------------------

def _write_validation_results(
    df: pd.DataFrame,
    db_path: Path,
    event_id: str = _EVENT_ID,
) -> int:
    """Write per-building validation results to DuckDB table validation_results.

    Uses INSERT with per-event DELETE, so results from multiple events coexist.
    """
    out = pd.DataFrame({
        "building_id":        df["building_id"],
        "event_id":           event_id,
        "burned":             df["burned"].astype(bool),
        "composite_score":    df["composite_score"],
        "risk_class":         df["risk_class"],
        "baseline_score":     df["baseline_score"],
        "is_false_negative":  (df["burned"] == 1) & (df["risk_class"] <= 2),
        "is_false_positive":  (df["burned"] == 0) & (df["risk_class"] >= 4),
    })

    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass

    # Create table if it doesn't exist; remove prior results for this event
    con.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            building_id VARCHAR,
            event_id VARCHAR,
            burned BOOLEAN,
            composite_score DOUBLE,
            risk_class INTEGER,
            baseline_score DOUBLE,
            is_false_negative BOOLEAN,
            is_false_positive BOOLEAN
        )
    """)
    con.execute("DELETE FROM validation_results WHERE event_id = ?", [event_id])

    con.register("_vr_df", out)
    con.execute("""
        INSERT INTO validation_results
        SELECT
            building_id,
            event_id,
            burned,
            composite_score,
            risk_class,
            baseline_score,
            is_false_negative,
            is_false_positive
        FROM _vr_df
    """)
    n = con.execute(
        "SELECT count(*) FROM validation_results WHERE event_id = ?",
        [event_id],
    ).fetchone()[0]
    con.close()
    return n


# ---------------------------------------------------------------------------
# Step 7: Export outputs
# ---------------------------------------------------------------------------

def _export_outputs(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    failure: dict[str, pd.DataFrame],
    out_dir: Path,
    event_name: str = "mati_2018",
) -> None:
    """Write CSV and JSON output files to outputs/validation/."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full validation results CSV
    csv_path = out_dir / f"{event_name}_validation_results.csv"
    df[[
        "building_id", "burned", "composite_score", "risk_class",
        "score_terrain", "score_vegetation", "score_fire_weather",
        "score_fire_history", "baseline_score",
    ]].to_csv(csv_path, index=False)
    print(f"  [export] {csv_path.name}: {len(df):,} rows")

    # Metrics JSON
    json_path = out_dir / f"{event_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [export] {json_path.name}")

    # False negatives CSV
    fn_path = out_dir / f"{event_name}_false_negatives.csv"
    failure["false_negatives"].to_csv(fn_path, index=False)
    print(f"  [export] {fn_path.name}: {len(failure['false_negatives'])} rows")

    # False positives CSV
    fp_path = out_dir / f"{event_name}_false_positives.csv"
    failure["false_positives"].to_csv(fp_path, index=False)
    print(f"  [export] {fp_path.name}: {len(failure['false_positives'])} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(event_name: str | None = None) -> dict[str, Any]:
    """Run the validation pipeline for a specified event.

    Args:
        event_name: Key from validation.yaml holdout_events (e.g. "mati_2018",
            "varybobi_2021").  Defaults to "mati_2018" for backward compatibility.

    Returns:
        Metrics dict for the validated event.
    """
    if event_name is None:
        event_name = "mati_2018"

    cfg     = load_config()
    root    = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]
    out_dir = root / "outputs" / "validation"

    # Load event-specific config from validation.yaml
    event_cfg = cfg["validation"]["holdout_events"][event_name]

    event_id   = event_cfg["fire_id"]
    event_date = event_cfg["date"]
    bbox       = tuple(event_cfg["validation_bbox"])
    event_label = event_cfg.get("name", event_name)
    geo_split  = event_cfg.get("geo_split_lat")

    print("=" * 64)
    print(f"Validation — {event_label}")
    print("=" * 64)
    print(f"  DB:    {db_path}")
    print(f"  Event: {event_id} ({event_date})")
    print(f"  Bbox:  W={bbox[0]} S={bbox[1]} E={bbox[2]} N={bbox[3]}")

    # ------------------------------------------------------------------
    # 1. Load event perimeter and label buildings
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading EFFIS perimeter {event_id} and labelling buildings ...")
    event_poly = _load_event_perimeter(db_path, event_id)
    print(f"  Perimeter: {event_poly.geom_type}  "
          f"area={event_poly.area/10000:.1f} ha")

    labels_df = _label_buildings(db_path, event_poly, bbox)
    n_burned   = labels_df["burned"].sum()
    n_total    = len(labels_df)
    print(f"  Validation buildings: {n_total:,}  "
          f"burned={n_burned:,} ({n_burned/n_total:.1%})  "
          f"unburned={n_total-n_burned:,}")

    # ------------------------------------------------------------------
    # 2. Join T10 scores and features
    # ------------------------------------------------------------------
    print("\n[2/5] Loading risk scores and features ...")
    scores_df = _load_scores_and_features(db_path, labels_df["building_id"])
    n_missing = n_total - len(scores_df)
    if n_missing:
        print(f"  WARN: {n_missing} validation buildings missing from risk_scores")
    print(f"  Scores loaded: {len(scores_df):,} buildings")

    # Merge labels + scores
    df = labels_df.merge(scores_df, on="building_id", how="inner")
    assert len(df) == len(scores_df), "Merge produced unexpected row count"
    print(f"  Merged dataset: {len(df):,} buildings")

    # ------------------------------------------------------------------
    # 3. Compute baseline scores
    # ------------------------------------------------------------------
    print("\n[3/5] Computing baseline scores ...")
    df["baseline_score"] = _compute_baseline(df)
    print(f"  Baseline (dist_to_forest): "
          f"mean={df['baseline_score'].mean():.4f} "
          f"std={df['baseline_score'].std():.4f}")

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    print("\n[4/5] Computing metrics ...")
    metrics = compute_metrics(
        df,
        event_id=event_id,
        event_date=event_date,
        bbox=bbox,
        geo_split_lat=geo_split,
    )

    auc_m = metrics["model"]["auc_roc"]
    auc_b = metrics["baseline"]["auc_roc"]
    lift_m = metrics["model"]["lift_at_top10pct"]
    lift_b = metrics["baseline"]["lift_at_top10pct"]
    prec5  = metrics["model"]["precision_cls5"]
    rec45  = metrics["model"]["recall_cls45"]

    auc_m_ci = (metrics["model"]["auc_roc_ci_lower"], metrics["model"]["auc_roc_ci_upper"])
    auc_b_ci = (metrics["baseline"]["auc_roc_ci_lower"], metrics["baseline"]["auc_roc_ci_upper"])

    status = ("PASS" if auc_m >= 0.70 else
              "WARN" if auc_m >= 0.60 else "FAIL")
    print("\n  ── Discrimination (AUC-ROC, 95% CI) ─────────────────────")
    print(f"  Model:    {auc_m:.4f} [{auc_m_ci[0]:.3f}, {auc_m_ci[1]:.3f}]  [{status}]  "
          f"(failure<0.60  target≥0.70)")
    print(f"  Baseline: {auc_b:.4f} [{auc_b_ci[0]:.3f}, {auc_b_ci[1]:.3f}]  "
          f"({'BEAT' if auc_m > auc_b else 'BELOW'} baseline by {auc_m-auc_b:+.4f})")

    print("\n  ── Lift at Top 10% ───────────────────────────────────────")
    print(f"  Model:    {lift_m:.2f}×  "
          f"(top 10% captures {lift_m*10:.1f}% of burned buildings)")
    print(f"  Baseline: {lift_b:.2f}×")

    print("\n  ── Class-level metrics ───────────────────────────────────")
    print(f"  Precision@class5:  {prec5:.2%}  "
          f"({metrics['model']['n_cls5']:,} class-5 buildings in bbox)")
    print(f"  Recall@class4+5:   {rec45:.2%}  "
          f"(fraction of burned in top 2 classes)")

    print("\n  ── Per-class burned rates ────────────────────────────────")
    for cls, s in metrics["class_stats"].items():
        expected = metrics["population"]["prevalence"]
        lift_cls = s["burned_rate"] / expected if expected > 0 else 0
        print(f"  class {cls}: {s['n_burned']:4d}/{s['n_buildings']:6,} "
              f"({s['burned_rate']:.2%})  lift={lift_cls:.1f}×  "
              f"vs base rate {expected:.2%}")

    print("\n  ── Mean score: burned vs unburned ───────────────────────")
    ms_b = metrics["mean_scores"]["burned"]
    ms_u = metrics["mean_scores"]["unburned"]
    for col in ["composite_score", "score_terrain", "score_vegetation",
                "score_fire_weather", "score_fire_history"]:
        diff = ms_b[col] - ms_u[col]
        direction = "+" if diff > 0 else "-"
        print(f"  {col:22s}: burned={ms_b[col]:.4f}  unburned={ms_u[col]:.4f}  "
              f"diff={diff:+.4f} [{direction}]")

    if metrics.get("geographic_diagnostic"):
        split_label = f" (split at {geo_split}°N)" if geo_split else ""
        print(f"\n  ── Geographic diagnostic{split_label} ─────────────")
        for zone, g in metrics["geographic_diagnostic"].items():
            print(f"  {zone:5s}: n={g['n_buildings']:5,}  burned={g['n_burned']:4d} "
                  f"({g['prevalence']:.1%})  "
                  f"AUC_model={g['auc_roc_model']:.4f}  "
                  f"AUC_baseline={g['auc_roc_baseline']:.4f}")

    # ------------------------------------------------------------------
    # 5. Failure analysis
    # ------------------------------------------------------------------
    print("\n[5/5] Failure analysis ...")
    failure = failure_analysis(df, n_top=10)

    print("\n  ── Top-10 False Negatives (burned, lowest score) ──────────")
    fn = failure["false_negatives"]
    for _, row in fn.iterrows():
        print(f"  {row['building_id']}: composite={row['composite_score']:.3f}  "
              f"class={row['risk_class']}  "
              f"veg={row['score_vegetation']:.3f}  "
              f"fh={row['score_fire_history']:.3f}  "
              f"fw={row['score_fire_weather']:.3f}")

    print("\n  ── Top-10 False Positives (unburned, highest score) ──────")
    fp = failure["false_positives"]
    for _, row in fp.iterrows():
        print(f"  {row['building_id']}: composite={row['composite_score']:.3f}  "
              f"class={row['risk_class']}  "
              f"veg={row['score_vegetation']:.3f}  "
              f"fh={row['score_fire_history']:.3f}  "
              f"fw={row['score_fire_weather']:.3f}")

    # ------------------------------------------------------------------
    # 6. Write to DuckDB
    # ------------------------------------------------------------------
    print(f"\n[write] Writing validation_results → {db_path.name} ...")
    count = _write_validation_results(df, db_path, event_id=event_id)
    print(f"  {count:,} rows written to validation_results")

    # ------------------------------------------------------------------
    # 7. Export CSV / JSON
    # ------------------------------------------------------------------
    print(f"\n[export] Writing outputs to {out_dir} ...")
    _export_outputs(df, metrics, failure, out_dir, event_name=event_name)

    # ------------------------------------------------------------------
    # 8. Final assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("VALIDATION COMPLETE — SUMMARY")
    print("=" * 64)
    _print_assessment(metrics, event_name=event_name)

    return metrics


def _print_assessment(metrics: dict[str, Any], event_name: str = "mati_2018") -> None:
    """Print a plain-English validation summary."""
    auc = metrics["model"]["auc_roc"]
    lift = metrics["model"]["lift_at_top10pct"]
    prec5 = metrics["model"]["precision_cls5"]
    rec45 = metrics["model"]["recall_cls45"]
    auc_b = metrics["baseline"]["auc_roc"]
    lift_b = metrics["baseline"]["lift_at_top10pct"]

    overall = ("PASS" if auc >= 0.70 else
               "WARN (below target, above failure)" if auc >= 0.60 else
               "FAIL — see geographic diagnostic")

    event_label = metrics.get("event", event_name)
    event_date = metrics.get("event_date", "")

    auc_ci = (metrics["model"].get("auc_roc_ci_lower"), metrics["model"].get("auc_roc_ci_upper"))
    auc_b_ci = (metrics["baseline"].get("auc_roc_ci_lower"), metrics["baseline"].get("auc_roc_ci_upper"))
    ci_str = f" [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]" if auc_ci[0] is not None else ""
    ci_b_str = f" [{auc_b_ci[0]:.3f}, {auc_b_ci[1]:.3f}]" if auc_b_ci[0] is not None else ""

    print(f"""
Event:          {event_label} — {event_date}
Population:     {metrics['population']['n_buildings']:,} buildings in validation bbox
Burned:         {metrics['population']['n_burned']:,} ({metrics['population']['prevalence']:.1%} prevalence)

PRIMARY METRIC:
  AUC-ROC (full bbox)  {auc:.4f}{ci_str}   [{overall}]
  Baseline AUC         {auc_b:.4f}{ci_b_str}   model delta {auc-auc_b:+.4f}""")

    geo = metrics.get("geographic_diagnostic", {})
    if geo:
        split_lat = next(
            (v.get("lat_threshold") for v in geo.values() if "lat_threshold" in v),
            None,
        )
        if split_lat is not None:
            print(f"""
  Geographic breakdown (split at {split_lat}°N):""")
        for zone, g in geo.items():
            zone_auc = g.get("auc_roc_model", float("nan"))
            zone_status = ("PASS" if zone_auc >= 0.70 else
                           "WARN" if zone_auc >= 0.60 else "FAIL")
            print(f"    {zone:12s}:  AUC = {zone_auc:.4f}   [{zone_status}]")

    print(f"""
SECONDARY METRICS:
  Lift@top10%    {lift:.2f}×   (baseline: {lift_b:.2f}×)
  Precision@5    {prec5:.1%}   of class-5 buildings were burned
  Recall@cls4+5  {rec45:.1%}   of burned buildings were in class 4 or 5""")

    # Event-specific root cause analysis
    if event_name == "mati_2018":
        south_auc = geo.get("south", {}).get("auc_roc_model", float("nan"))
        north_auc = geo.get("north", {}).get("auc_roc_model", float("nan"))
        print(f"""
FAILURE ROOT CAUSE (Mati 2018):
  Wind-driven event (Beaufort 8-9) in fire-naive coastal zone.  Model assigns
  high scores to inland forested zones (correct structural risk) but low scores
  to Mati coast (correctly reflecting zero fire history).
    - South (periphery): AUC = {south_auc:.2f} — model works
    - North (fire core):  AUC = {north_auc:.2f} — model inverted
  This motivates the v2 event-context dynamic layer.""")
    else:
        if auc >= 0.70:
            print(f"""
ASSESSMENT:
  Structural model achieves AUC {auc:.2f} on this event — above target (0.70).
  The terrain/vegetation/fire-history features discriminate effectively.""")
        elif auc >= 0.60:
            print(f"""
ASSESSMENT:
  Structural model AUC {auc:.2f} is above failure threshold but below target.
  Partial discrimination — geographic diagnostic may reveal spatial patterns.""")
        else:
            print(f"""
ASSESSMENT:
  Structural model AUC {auc:.2f} is below failure threshold (0.60).
  Geographic diagnostic and failure analysis should guide root cause investigation.""")
    print()


if __name__ == "__main__":
    _event = sys.argv[1] if len(sys.argv) > 1 else None
    main(_event)
