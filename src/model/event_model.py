"""
src/model/event_model.py
-------------------------
LightGBM event-context model with leave-one-event-out (LOEO) cross-validation
and SHAP explainability.

Architecture
------------
  v1 structural features (21) + v2 dynamic features (5) → LightGBM binary
  classifier → P(burned) per building per event.

  Leave-one-event-out CV (N events from config/validation.yaml):
    - For each event: train on all other events, predict the held-out one
    - Tests generalization across fire types (wind-driven, terrain-driven, etc.)

  SHAP: TreeExplainer on full-data model for global feature importance.

Inputs
------
  DuckDB tables:
    features_terrain, features_vegetation, features_fire_weather,
    features_fire_history       — 21 structural features per building
    features_dynamic_event      — 5 dynamic features per building per event
    validation_results          — burned label + v1 composite_score per building per event

  Config:
    config/validation.yaml      — event definitions (fire_id ↔ event_name mapping)

Outputs
-------
  DuckDB table ``model_v2_predictions``:
    event_id, building_id, lgbm_prob, structural_score

  PNG plots in outputs/validation/:
    shap_importance.png         — global bar chart (mean |SHAP|)
    shap_beeswarm.png          — beeswarm dot plot

  JSON:
    outputs/validation/v2_model_metrics.json — per-event AUC comparison

Limitations
-----------
  - 4-event LOEO is minimally viable; expanding to 6+ requires AOI extension.
  - LightGBM uses near-default hyperparameters (no tuning by design).
  - Dynamic features share the same ERA5 grid cell value across ~1000 buildings
    per cell (9 km resolution).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_path
from src.validation.validator import bootstrap_auc_ci

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

STRUCTURAL_FEATURES = [
    # Terrain (5)
    "elevation_m", "slope_deg", "south_aspect_score", "tpi_300m", "tri_300m",
    # Vegetation (6)
    "veg_fraction_100m", "veg_fraction_500m", "dist_to_forest_m",
    "dist_to_scrubland_m", "wui_class", "veg_continuity_500m",
    # Fire weather climatology (5)
    "fwi_season_mean", "fwi_season_p90", "fwi_season_max",
    "dc_season_mean", "fwi_extreme_days",
    # Fire history (5)
    "dist_to_nearest_fire_m", "fire_count_5km", "fire_count_10km",
    "firms_hotspot_count_5km", "recency_score",
]

DYNAMIC_FEATURES = [
    "wind_speed_max_24h", "wind_dir_consistency",
    "vpd_event_day", "dc_antecedent_30d", "fwi_event_day",
]

ALL_FEATURES = STRUCTURAL_FEATURES + DYNAMIC_FEATURES

# Fire history features contain temporal leakage: they include the test event's
# own fire perimeter and hotspots (fire_history_cutoff not yet applied as a
# temporal filter in fire_history.py).  Exclude from LOEO evaluation to prevent
# the model from trivially matching "near the fire that already happened."
# Full temporal-cutoff implementation is deferred to v3.
FIRE_HISTORY_FEATURES = [
    "dist_to_nearest_fire_m", "fire_count_5km", "fire_count_10km",
    "firms_hotspot_count_5km", "recency_score",
]

LOEO_FEATURES = [f for f in ALL_FEATURES if f not in FIRE_HISTORY_FEATURES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_event_map(cfg: dict) -> dict[str, dict]:
    """Return {event_name: {fire_id, date, ...}} from validation config."""
    return {
        name: {**ev, "event_name": name}
        for name, ev in cfg["validation"]["holdout_events"].items()
    }


def load_event_dataset(
    db_path: Path,
    event_name: str,
    fire_id: str,
) -> pd.DataFrame:
    """Load structural + dynamic features + burned label for one event.

    Joins across 6 DuckDB tables on building_id.  The event_name key is used
    for ``features_dynamic_event`` while fire_id is used for
    ``validation_results`` (different naming conventions from v1 vs v2 code).

    Returns:
        DataFrame with building_id, burned, composite_score, 26 feature columns.
    """
    structural_cols = ", ".join(
        [f"t.{c}" for c in STRUCTURAL_FEATURES[:5]]
        + [f"v.{c}" for c in STRUCTURAL_FEATURES[5:11]]
        + [f"w.{c}" for c in STRUCTURAL_FEATURES[11:16]]
        + [f"h.{c}" for c in STRUCTURAL_FEATURES[16:21]]
    )
    dynamic_cols = ", ".join(f"d.{c}" for c in DYNAMIC_FEATURES)

    query = f"""
        SELECT
            vr.building_id,
            vr.burned::INTEGER AS burned,
            vr.composite_score,
            {structural_cols},
            {dynamic_cols}
        FROM validation_results vr
        JOIN features_terrain        t USING (building_id)
        JOIN features_vegetation     v USING (building_id)
        JOIN features_fire_weather   w USING (building_id)
        JOIN features_fire_history   h USING (building_id)
        JOIN features_dynamic_event  d ON d.building_id = vr.building_id
                                      AND d.event_id = ?
        WHERE vr.event_id = ?
    """  # noqa: S608

    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(query, [event_name, fire_id]).df()
    con.close()

    if len(df) == 0:
        raise ValueError(
            f"No data for event {event_name} (fire_id={fire_id}). "
            "Check that validation_results and features_dynamic_event are populated."
        )

    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

_LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,
    "random_state": 42,
    "verbose": -1,
    "importance_type": "gain",
}


def train_loeo(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str] = ALL_FEATURES,
) -> dict[str, dict]:
    """Leave-one-event-out cross-validation.

    For each event, trains on ALL other events and predicts the held-out event.

    Returns:
        {event_name: {"model", "y_true", "y_prob", "X_test", "df_test"}}
    """
    event_names = list(datasets.keys())
    results: dict[str, dict] = {}

    for test_event in event_names:
        train_events = [e for e in event_names if e != test_event]
        train_df = pd.concat(
            [datasets[e] for e in train_events], ignore_index=True,
        )

        X_train = train_df[feature_cols]
        y_train = train_df["burned"]
        X_test = datasets[test_event][feature_cols]
        y_test = datasets[test_event]["burned"]

        model = lgb.LGBMClassifier(**_LGB_PARAMS)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        results[test_event] = {
            "model": model,
            "y_true": y_test.values,
            "y_prob": y_prob,
            "X_test": X_test,
            "df_test": datasets[test_event],
        }

    return results


def train_full(
    datasets: dict[str, pd.DataFrame],
    feature_cols: list[str] = ALL_FEATURES,
) -> lgb.LGBMClassifier:
    """Train on all events combined (for SHAP global explanation)."""
    full_df = pd.concat(datasets.values(), ignore_index=True)
    X = full_df[feature_cols]
    y = full_df["burned"]

    model = lgb.LGBMClassifier(**_LGB_PARAMS)
    model.fit(X, y)
    return model



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(loeo_results: dict, datasets: dict) -> dict:
    """Compute AUC-ROC for v1 (structural) and v2 (LightGBM LOEO) per event.

    LOEO is the only honest evaluation metric for event-level generalization.
    In-sample and pooled-CV metrics are intentionally omitted because
    spatial autocorrelation inflates AUC for geographically clustered
    building data (nearby buildings share features and burn outcomes).
    """
    metrics: dict[str, dict] = {}

    for event_name, res in loeo_results.items():
        y_true = res["y_true"]
        y_prob_v2 = res["y_prob"]
        structural_score = datasets[event_name]["composite_score"].values

        auc_v1, v1_ci_lo, v1_ci_hi = bootstrap_auc_ci(
            y_true, structural_score, n_bootstraps=1000, ci=0.95,
        )
        auc_v2, v2_ci_lo, v2_ci_hi = bootstrap_auc_ci(
            y_true, y_prob_v2, n_bootstraps=1000, ci=0.95,
        )
        n_burned = int(y_true.sum())
        n_total = len(y_true)

        metrics[event_name] = {
            "n_buildings": n_total,
            "n_burned": n_burned,
            "prevalence": round(n_burned / n_total, 4),
            "auc_structural_v1": round(auc_v1, 4),
            "auc_structural_v1_ci_lower": round(v1_ci_lo, 4),
            "auc_structural_v1_ci_upper": round(v1_ci_hi, 4),
            "auc_lgbm_v2": round(auc_v2, 4),
            "auc_lgbm_v2_ci_lower": round(v2_ci_lo, 4),
            "auc_lgbm_v2_ci_upper": round(v2_ci_hi, 4),
            "auc_delta": round(auc_v2 - auc_v1, 4),
        }

    return metrics


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    output_dir: Path,
    max_display: int = 15,
) -> None:
    """Generate SHAP importance bar + beeswarm plots.

    Uses TreeExplainer for efficiency with LightGBM.
    Saves to output_dir/shap_importance.png and shap_beeswarm.png.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar plot — global feature importance (mean |SHAP|)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
    ax.set_title("SHAP Feature Importance (mean |SHAP value|)")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'shap_importance.png'}")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm — Feature Impact on P(burned)")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_beeswarm.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'shap_beeswarm.png'}")


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def _write_predictions(
    loeo_results: dict,
    datasets: dict,
    db_path: Path,
) -> int:
    """Write LOEO predictions to DuckDB table model_v2_predictions."""
    rows = []
    for event_name, res in loeo_results.items():
        df_test = res["df_test"]
        rows.append(pd.DataFrame({
            "event_id": event_name,
            "building_id": df_test["building_id"].values,
            "lgbm_prob": res["y_prob"],
            "structural_score": df_test["composite_score"].values,
        }))

    pred_df = pd.concat(rows, ignore_index=True)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET storage_compatibility_level = 'latest';")
    except Exception:
        pass

    con.execute("DROP TABLE IF EXISTS model_v2_predictions")
    con.register("_pred_df", pred_df)
    con.execute("""
        CREATE TABLE model_v2_predictions AS
        SELECT event_id, building_id, lgbm_prob, structural_score
        FROM _pred_df
    """)
    n = con.execute("SELECT count(*) FROM model_v2_predictions").fetchone()[0]
    con.close()
    return n


def _write_metrics(metrics: dict, output_dir: Path) -> Path:
    """Write metrics JSON to outputs/validation/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "v2_model_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run LightGBM LOEO + SHAP for all validation events."""
    cfg = load_config()
    root = resolve_path(".")
    db_path = root / cfg["pipeline"]["paths"]["db"]
    val_dir = resolve_path("outputs/validation")

    event_map = _build_event_map(cfg)

    # ── 1. Load datasets ─────────────────────────────────────────────────
    print("[event_model] Loading datasets ...")
    datasets: dict[str, pd.DataFrame] = {}
    for name, ev in event_map.items():
        df = load_event_dataset(db_path, name, ev["fire_id"])
        datasets[name] = df
        print(
            f"  {name}: {len(df):,} buildings, "
            f"{int(df['burned'].sum()):,} burned "
            f"({df['burned'].mean():.1%}), "
            f"{len(ALL_FEATURES)} features"
        )

    # ── 2. LOEO cross-validation ─────────────────────────────────────────
    print(
        f"\n[event_model] Leave-one-event-out CV "
        f"({len(LOEO_FEATURES)} features, fire history excluded) ..."
    )
    loeo = train_loeo(datasets, feature_cols=LOEO_FEATURES)

    # ── 3. Metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(loeo, datasets)

    print("\n  ┌─────────────────┬──────────────────────┬──────────────────────┬────────┐")
    print("  │ Event           │ AUC v1 [95% CI]      │ AUC v2 [95% CI]      │  Δ AUC │")
    print("  ├─────────────────┼──────────────────────┼──────────────────────┼────────┤")
    for name, m in metrics.items():
        v1 = m["auc_structural_v1"]
        v1_ci = f"[{m['auc_structural_v1_ci_lower']:.2f}, {m['auc_structural_v1_ci_upper']:.2f}]"
        v2 = m["auc_lgbm_v2"]
        v2_ci = f"[{m['auc_lgbm_v2_ci_lower']:.2f}, {m['auc_lgbm_v2_ci_upper']:.2f}]"
        delta = m["auc_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"  │ {name:<15s} │ {v1:.4f} {v1_ci:<12s} │ {v2:.4f} {v2_ci:<12s} │ {sign}{delta:.4f} │")
    print("  └─────────────────┴──────────────────────┴──────────────────────┴────────┘")
    print("  (LOEO: train on all other events, predict held-out event)")

    # ── 4. Full-data model + SHAP ─────────────────────────────────────────
    print("\n[event_model] Training full-data model for SHAP ...")
    full_model = train_full(datasets)

    full_X = pd.concat(
        [ds[ALL_FEATURES] for ds in datasets.values()], ignore_index=True,
    )
    print("[event_model] Computing SHAP values ...")
    compute_shap(full_model, full_X, val_dir)

    # ── 5. Write outputs ──────────────────────────────────────────────────
    metrics["_meta"] = {
        "loeo_n_features": len(LOEO_FEATURES),
        "shap_n_features": len(ALL_FEATURES),
        "fire_history_excluded_from_loeo": True,
        "is_unbalance": True,
        "note": (
            "Fire history features excluded from LOEO to avoid temporal "
            "leakage (test event fire included in feature computation). "
            "Full temporal cutoff deferred to v3."
        ),
    }

    n_pred = _write_predictions(loeo, datasets, db_path)
    print(f"\n[event_model] Written {n_pred:,} rows to model_v2_predictions")

    metrics_path = _write_metrics(metrics, val_dir)
    print(f"[event_model] Metrics: {metrics_path}")

    print("\n[event_model] Done.")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
