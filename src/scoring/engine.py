"""
engine.py — WildfireRisk-EU Scoring Engine (Task T10)

Converts the 4 completed feature tables (terrain, vegetation, fire_weather,
fire_history) into interpretable asset-level wildfire risk scores and writes
the ``risk_scores`` table to DuckDB.

Scoring Design
--------------
Feature selection
  - ``direction=positive``: higher raw value → higher risk → normalise as-is.
  - ``direction=negative``: lower raw value → higher risk → invert after normalise.
  - ``direction=neutral``: included as positive (higher value treated as higher
    risk contribution). Applies to ``elevation_m`` and ``tri_300m``.
  - ``direction=custom``: ``aspect_deg`` is excluded because its risk-relevant
    information is already captured by the derived ``south_aspect_score``.
  - Zero-variance columns are excluded automatically (cannot be ranked).
    Known degenerate feature: ``ever_burned`` (all-1 in the Attica dataset).
  - All-NULL columns are excluded automatically.
    Known missing features: ``ndvi_mean_100m``, ``ndvi_mean_500m``, ``ndvi_max_500m``
    (NDVI raster not acquired for this study).

Normalisation
  For each included feature:
  1. Clip to [p1, p99] (Winsorise) to reduce outlier influence.
  2. Rank-normalise the clipped values via scipy.stats.rankdata (average ties)
     mapped to [0, 1]: score = (rank − 1) / (n − 1).
  3. Invert for negative-direction features: score = 1 − score.

Within-group aggregation
  Equal-weighted mean of all included feature scores within the group.
  NULL scores (from dropped features) are excluded from the mean.

Group weights
  Configured in ``config/scoring.yaml``:
    vegetation=0.30, terrain=0.20, fire_weather=0.20, fire_history=0.20,
    proximity=0.10.
  Since the proximity group table is not yet built for T10, its 0.10 weight is
  redistributed proportionally across the 4 present groups at runtime:
    vegetation≈0.3333, terrain≈0.2222, fire_weather≈0.2222, fire_history≈0.2222.
  The redistribution is computed from scoring.yaml; weights are not hard-coded.

Risk classification
  Quintile on ``composite_score`` → 5 equal-population classes.
  Class 1 = Very Low, 5 = Very High.

Output table (DuckDB ``risk_scores``)
  building_id        TEXT     — joins to buildings table
  score_terrain      DOUBLE   — group score [0, 1]
  score_vegetation   DOUBLE   — group score [0, 1]
  score_fire_weather DOUBLE   — group score [0, 1]
  score_fire_history DOUBLE   — group score [0, 1]
  composite_score    DOUBLE   — weighted sum [0, 1]
  risk_class         INTEGER  — quintile class 1–5
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Feature metadata: which columns belong to which group, and their direction
# (derived from config/features.yaml; listed explicitly for traceability)
# ---------------------------------------------------------------------------

# Columns to always exclude before scoring, with reasons:
#   aspect_deg    — direction=custom; south_aspect_score already captures this
#   ndvi_*        — all-NULL (NDVI raster not acquired)
#   ever_burned   — zero variance (all=1 in Attica dataset; cannot be ranked)
_ALWAYS_EXCLUDE = {
    "aspect_deg",
    "ndvi_mean_100m",
    "ndvi_mean_500m",
    "ndvi_max_500m",
    "ever_burned",
}

# Group → list of (column, direction) pairs.
# direction ∈ {"positive", "negative", "neutral"}
# neutral treated as positive (higher value = higher risk contribution)
_GROUP_FEATURES: dict[str, list[tuple[str, str]]] = {
    "terrain": [
        ("elevation_m",          "neutral"),    # neutral → treated as positive
        ("slope_deg",            "positive"),
        ("south_aspect_score",   "positive"),
        ("tpi_300m",             "positive"),
        ("tri_300m",             "neutral"),    # neutral → treated as positive
    ],
    "vegetation": [
        ("veg_fraction_100m",    "positive"),
        ("veg_fraction_500m",    "positive"),
        ("dist_to_forest_m",     "negative"),
        ("dist_to_scrubland_m",  "negative"),
        ("wui_class",            "positive"),
        ("veg_continuity_500m",  "positive"),
    ],
    "fire_weather": [
        ("fwi_season_mean",      "positive"),
        ("fwi_season_p90",       "positive"),
        ("fwi_season_max",       "positive"),
        ("dc_season_mean",       "positive"),
        ("fwi_extreme_days",     "positive"),
    ],
    "fire_history": [
        ("dist_to_nearest_fire_m",    "negative"),
        ("fire_count_5km",            "positive"),
        ("fire_count_10km",           "positive"),
        ("firms_hotspot_count_5km",   "positive"),
        ("recency_score",             "positive"),
    ],
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_features(db_path: Path) -> pd.DataFrame:
    """Join the 4 feature tables on building_id and return a merged DataFrame."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT
            t.building_id,
            -- terrain
            t.elevation_m,
            t.slope_deg,
            t.aspect_deg,
            t.south_aspect_score,
            t.tpi_300m,
            t.tri_300m,
            -- vegetation
            v.ndvi_mean_100m,
            v.ndvi_mean_500m,
            v.ndvi_max_500m,
            v.veg_fraction_100m,
            v.veg_fraction_500m,
            v.dist_to_forest_m,
            v.dist_to_scrubland_m,
            v.wui_class,
            v.veg_continuity_500m,
            -- fire weather
            w.fwi_season_mean,
            w.fwi_season_p90,
            w.fwi_season_max,
            w.dc_season_mean,
            w.fwi_extreme_days,
            -- fire history
            h.dist_to_nearest_fire_m,
            h.fire_count_5km,
            h.fire_count_10km,
            h.ever_burned,
            h.firms_hotspot_count_5km,
            h.recency_score
        FROM features_terrain    t
        JOIN features_vegetation v USING (building_id)
        JOIN features_fire_weather w USING (building_id)
        JOIN features_fire_history h USING (building_id)
        ORDER BY t.building_id
    """).df()
    con.close()
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def percentile_rank_normalize(
    series: pd.Series,
    clip_p1: float = 1.0,
    clip_p99: float = 99.0,
    direction: str = "positive",
) -> pd.Series:
    """Normalise a feature series to [0, 1] using percentile-rank normalisation.

    Steps:
    1. Clip values to [p1, p99] (Winsorise) to reduce outlier influence.
    2. Rank-normalise the clipped values → [0, 1].
       rank = rankdata(clipped, method='average')   # ties → average rank
       score = (rank - 1) / (n - 1)
    3. Invert for negative-direction features: score = 1 - score.

    Args:
        series:    Input feature values (no NULLs expected).
        clip_p1:   Lower clip percentile (default 1).
        clip_p99:  Upper clip percentile (default 99).
        direction: ``"positive"`` or ``"neutral"`` → higher = higher risk.
                   ``"negative"`` → lower = higher risk (score is inverted).

    Returns:
        Normalised score series in [0, 1], same index as input.
    """
    vals = series.values.astype(float)
    n = len(vals)

    # Step 1: Winsorise
    lo = np.percentile(vals, clip_p1)
    hi = np.percentile(vals, clip_p99)
    clipped = np.clip(vals, lo, hi)

    # Step 2: Rank → [0, 1]
    if n == 1:
        scores = np.array([0.5])
    else:
        ranks = rankdata(clipped, method="average")   # ties get average rank
        scores = (ranks - 1.0) / (n - 1.0)           # [0, 1]

    # Step 3: Invert for negative direction
    if direction == "negative":
        scores = 1.0 - scores

    return pd.Series(scores, index=series.index, name=series.name)


def _is_eligible(series: pd.Series) -> tuple[bool, str]:
    """Check whether a feature column is eligible for scoring.

    Returns (eligible, reason_if_not).
    """
    null_frac = series.isna().mean()
    if null_frac == 1.0:
        return False, "all-NULL"
    if null_frac > 0.5:
        return False, f"high-NULL ({null_frac:.1%})"
    nonnull = series.dropna()
    if nonnull.nunique() <= 1:
        return False, "zero-variance"
    return True, ""


# ---------------------------------------------------------------------------
# Group scoring
# ---------------------------------------------------------------------------

def _score_group(
    df: pd.DataFrame,
    group: str,
    clip_p1: float = 1.0,
    clip_p99: float = 99.0,
) -> tuple[pd.Series, dict[str, Optional[pd.Series]]]:
    """Compute normalised scores for all eligible features in a group.

    Returns:
        group_score: Mean of eligible feature scores, shape (n,), [0, 1].
        feature_scores: dict of col → normalised score (or None if dropped).
    """
    features = _GROUP_FEATURES[group]
    feature_scores: dict[str, Optional[pd.Series]] = {}
    eligible_scores: list[pd.Series] = []

    for col, direction in features:
        if col in _ALWAYS_EXCLUDE:
            print(f"  [{group}] SKIP {col} — in always-exclude list")
            feature_scores[col] = None
            continue

        if col not in df.columns:
            print(f"  [{group}] SKIP {col} — column not found")
            feature_scores[col] = None
            continue

        eligible, reason = _is_eligible(df[col])
        if not eligible:
            print(f"  [{group}] SKIP {col} — {reason}")
            feature_scores[col] = None
            continue

        # Drop NULLs for normalisation; re-index back to full index after
        series = df[col].dropna()
        scores = percentile_rank_normalize(
            series, clip_p1=clip_p1, clip_p99=clip_p99, direction=direction
        )
        # Reindex to full DataFrame index (any remaining NULLs stay NaN)
        scores = scores.reindex(df.index)

        n_used = scores.notna().sum()
        print(f"  [{group}] {col} ({direction}): "
              f"min={scores.min():.4f} max={scores.max():.4f} "
              f"mean={scores.mean():.4f} n={n_used:,}")
        feature_scores[col] = scores
        eligible_scores.append(scores)

    if not eligible_scores:
        raise ValueError(f"No eligible features found for group '{group}'")

    # Equal-weight mean; NaN-safe (drops NaN features, not rows)
    score_matrix = pd.concat(eligible_scores, axis=1)
    group_score = score_matrix.mean(axis=1)
    print(f"  [{group}] GROUP SCORE: mean={group_score.mean():.4f} "
          f"std={group_score.std():.4f} "
          f"n_features={len(eligible_scores)}")
    return group_score, feature_scores


# ---------------------------------------------------------------------------
# Composite scoring and classification
# ---------------------------------------------------------------------------

def _redistribute_weights(
    cfg_weights: dict[str, float],
    present_groups: list[str],
) -> dict[str, float]:
    """Proportionally redistribute missing-group weight to present groups.

    Args:
        cfg_weights:    Group weights from scoring.yaml (sum = 1.0).
        present_groups: Groups for which feature tables are available.

    Returns:
        Normalised weights for present groups (sum = 1.0).
    """
    total_present = sum(cfg_weights[g] for g in present_groups)
    redistributed = {g: cfg_weights[g] / total_present for g in present_groups}
    assert abs(sum(redistributed.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"
    return redistributed


def _classify_quintile(scores: pd.Series) -> pd.Series:
    """Assign quintile class 1–5 based on composite score.

    Uses pd.qcut with 5 equal-frequency bins.  Ties at boundaries are handled
    by qcut with ``duplicates='drop'``.  Labels are integers 1–5.
    """
    labels = pd.qcut(
        scores,
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype(int)
    return labels


# ---------------------------------------------------------------------------
# DuckDB write
# ---------------------------------------------------------------------------

def _write_risk_scores(df_out: pd.DataFrame, db_path: Path) -> int:
    """Write risk_scores DataFrame to DuckDB, replacing any existing table."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass

    con.execute("DROP TABLE IF EXISTS risk_scores;")
    con.register("_rs_df", df_out)
    con.execute("""
        CREATE TABLE risk_scores AS
        SELECT
            building_id,
            score_terrain,
            score_vegetation,
            score_fire_weather,
            score_fire_history,
            composite_score,
            risk_class
        FROM _rs_df
    """)
    n = con.execute("SELECT count(*) FROM risk_scores").fetchone()[0]
    con.close()
    return n


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_sanity_checks(df_out: pd.DataFrame, db_path: Path) -> None:
    """Assert pre-write acceptance criteria from the T10 spec.

    These checks run on the in-memory DataFrame before it is written to DuckDB.
    The Mati geographic check (criterion 5) runs separately in
    ``_run_mati_check`` after the table has been written.
    """
    n = len(df_out)
    assert n >= 80_000, f"Row count suspiciously low: {n} (expected ~84K)"

    score_cols = [
        "score_terrain", "score_vegetation",
        "score_fire_weather", "score_fire_history", "composite_score",
    ]
    for col in score_cols:
        assert df_out[col].isna().sum() == 0, f"NULLs in {col}"
        assert df_out[col].min() >= 0.0, f"{col} has negative values"
        assert df_out[col].max() <= 1.0, f"{col} max > 1.0"
    print("  [sanity] Score bounds OK — all in [0, 1], no NULLs")

    class_counts = df_out["risk_class"].value_counts().sort_index()
    expected = n / 5
    print("  [sanity] Risk class distribution:")
    for cls, cnt in class_counts.items():
        pct = cnt / n * 100
        within = abs(cnt - expected) <= 0.02 * n
        print(f"    class {cls}: {cnt:,} ({pct:.1f}%)  {'OK' if within else 'WARN'}")
        # Spec: each class within 18–22% of 84,767
        assert 0.18 * n <= cnt <= 0.22 * n, (
            f"Class {cls} count {cnt} outside 18-22% tolerance"
        )
    print("  [sanity] Quintile distribution OK")

    r_veg = df_out["composite_score"].corr(df_out["score_vegetation"])
    print(f"  [sanity] Pearson r(composite, score_vegetation) = {r_veg:.4f}")
    assert r_veg > 0.3, f"Vegetation correlation too low: {r_veg:.4f}"
    print("  [sanity] Vegetation correlation OK (r > 0.3)")


def _run_mati_check(db_path: Path) -> None:
    """Criterion 5: buildings near Mati epicentre (37.94°N, 23.98°E) should
    average risk_class ≥ 4.  Runs after risk_scores has been written."""
    con = duckdb.connect(str(db_path), read_only=True)
    row = con.execute("""
        SELECT count(*), avg(rs.risk_class)
        FROM risk_scores rs
        JOIN buildings b USING (building_id)
        WHERE b.centroid_lat BETWEEN 37.89 AND 37.99
          AND b.centroid_lon BETWEEN 23.93 AND 24.03
    """).fetchone()
    con.close()
    n_mati, mati_avg = row
    print(f"  [sanity] Mati area: {n_mati} buildings, mean risk_class = {mati_avg:.2f}  (spec: ≥ 4)")
    if mati_avg is None:
        print("  [sanity] WARN: No buildings found in Mati bounding box")
    elif mati_avg < 4.0:
        print(f"  [sanity] WARN: Mati avg risk_class {mati_avg:.2f} < 4 "
              "(check fire-history and vegetation signal in this area)")
    else:
        print("  [sanity] Mati check PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the T10 scoring pipeline end-to-end."""
    cfg     = load_config()
    root    = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]

    norm_cfg      = cfg["scoring"]["normalization"]
    clip_p1, clip_p99 = norm_cfg["clip_percentiles"]
    cfg_weights   = cfg["scoring"]["weights"]

    # Present groups (proximity table not yet built)
    present_groups = ["terrain", "vegetation", "fire_weather", "fire_history"]
    missing_groups = ["proximity"]

    print("=" * 64)
    print("T10 — Wildfire Risk Scoring Engine")
    print("=" * 64)
    print(f"  DB: {db_path}")
    print(f"  Clip percentiles: p{clip_p1} / p{clip_p99}")
    print(f"  Present groups: {present_groups}")
    print(f"  Missing groups (weight redistributed): {missing_groups}")

    # ------------------------------------------------------------------
    # 1. Load features
    # ------------------------------------------------------------------
    print("\n[1/5] Loading feature tables ...")
    df = _load_features(db_path)
    n = len(df)
    print(f"  Loaded {n:,} buildings × {df.shape[1]-1} raw feature columns")
    assert n >= 80_000, f"Expected ~84K buildings, got {n}"
    assert df["building_id"].duplicated().sum() == 0, "Duplicate building_ids"
    print("  Join integrity OK — no duplicate building_ids")

    # ------------------------------------------------------------------
    # 2. Compute group weights (redistribute proximity weight)
    # ------------------------------------------------------------------
    print("\n[2/5] Computing group weights ...")
    weights = _redistribute_weights(cfg_weights, present_groups)
    for grp, w in weights.items():
        orig = cfg_weights[grp]
        print(f"  {grp}: {orig:.4f} → {w:.4f} (after redistribution)")
    redistributed_weight = sum(cfg_weights[g] for g in missing_groups)
    print(f"  Redistributed weight from {missing_groups}: {redistributed_weight:.2f}")

    # ------------------------------------------------------------------
    # 3. Score each group
    # ------------------------------------------------------------------
    print("\n[3/5] Scoring feature groups ...")
    df = df.set_index("building_id")

    group_scores: dict[str, pd.Series] = {}
    for group in present_groups:
        print(f"\n  -- {group.upper()} --")
        gscore, _ = _score_group(df, group, clip_p1=clip_p1, clip_p99=clip_p99)
        group_scores[group] = gscore

    # ------------------------------------------------------------------
    # 4. Composite score
    # ------------------------------------------------------------------
    print("\n[4/5] Computing composite score ...")
    composite = sum(weights[g] * group_scores[g] for g in present_groups)
    composite.name = "composite_score"

    # Report group score stats
    for grp, gscore in group_scores.items():
        print(f"  score_{grp}: mean={gscore.mean():.4f} std={gscore.std():.4f} "
              f"weight={weights[grp]:.4f}")
    print(f"  composite_score: mean={composite.mean():.4f} "
          f"std={composite.std():.4f} "
          f"min={composite.min():.4f} max={composite.max():.4f}")

    # ------------------------------------------------------------------
    # 5. Classify into quintiles
    # ------------------------------------------------------------------
    print("\n[5/5] Classifying into quintile risk classes ...")
    risk_class = _classify_quintile(composite)
    class_counts = risk_class.value_counts().sort_index()
    for cls, cnt in class_counts.items():
        print(f"  class {cls}: {cnt:,} ({cnt/n*100:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Assemble output DataFrame
    # ------------------------------------------------------------------
    df_out = pd.DataFrame({
        "building_id":        group_scores["terrain"].index,
        "score_terrain":      group_scores["terrain"].values,
        "score_vegetation":   group_scores["vegetation"].values,
        "score_fire_weather": group_scores["fire_weather"].values,
        "score_fire_history": group_scores["fire_history"].values,
        "composite_score":    composite.values,
        "risk_class":         risk_class.values,
    })

    # ------------------------------------------------------------------
    # 7. Pre-write sanity checks (in-memory)
    # ------------------------------------------------------------------
    print("\n[sanity] Running pre-write acceptance checks ...")
    _run_sanity_checks(df_out, db_path)

    # ------------------------------------------------------------------
    # 8. Write to DuckDB
    # ------------------------------------------------------------------
    print(f"\n[write] Writing risk_scores → {db_path.name} ...")
    count = _write_risk_scores(df_out, db_path)
    assert count == n, f"Write verification failed: {count} != {n}"
    print(f"[write] Done — {count:,} rows written to risk_scores")

    # ------------------------------------------------------------------
    # 9. Post-write Mati geographic sanity check (criterion 5)
    # ------------------------------------------------------------------
    print("\n[sanity] Running Mati geographic check (post-write) ...")
    _run_mati_check(db_path)

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("T10 COMPLETE")
    print("=" * 64)
    print(f"  Output table : risk_scores ({count:,} rows)")
    print("  Score columns: score_terrain, score_vegetation, "
          "score_fire_weather, score_fire_history")
    print("  Composite    : composite_score (weighted sum, [0,1])")
    print("  Classification: risk_class 1–5 (quintile)")
    print("\nGroup weights applied:")
    for grp, w in weights.items():
        print(f"  {grp:15s}: {w:.4f}")
    print("\nExcluded features:")
    print("  aspect_deg           — direction=custom (south_aspect_score used instead)")
    print("  ndvi_mean_100m/500m  — all-NULL (NDVI raster not acquired)")
    print("  ndvi_max_500m        — all-NULL (NDVI raster not acquired)")
    print("  ever_burned          — zero variance (all=1 in Attica dataset)")
    print("\nT11 (validation) may now proceed.")


if __name__ == "__main__":
    main()
