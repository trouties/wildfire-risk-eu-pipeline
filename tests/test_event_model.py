"""
tests/test_event_model.py
---------------------------
Unit tests for src/model/event_model.py.

All tests use synthetic data — no real DuckDB, no CDS downloads.
Targets: train_loeo, train_full, compute_metrics, feature list consistency.
"""

import numpy as np
import pandas as pd
import pytest

from src.model.event_model import (
    ALL_FEATURES,
    DYNAMIC_FEATURES,
    FIRE_HISTORY_FEATURES,
    LOEO_FEATURES,
    STRUCTURAL_FEATURES,
    compute_metrics,
    train_full,
    train_loeo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_event(
    n: int = 500,
    burn_rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic event dataset with 26 features + labels.

    High-wind / high-VPD buildings are more likely to burn (to give
    the model a learnable signal).
    """
    rng = np.random.default_rng(seed)

    data = {"building_id": [f"b_{i:05d}" for i in range(n)]}

    # Structural features — random noise (weak signal)
    for feat in STRUCTURAL_FEATURES:
        data[feat] = rng.normal(50, 10, n)

    # Dynamic features — inject signal
    data["wind_speed_max_24h"] = rng.exponential(5, n)
    data["wind_dir_consistency"] = rng.uniform(0.5, 5, n)
    data["vpd_event_day"] = rng.uniform(10, 50, n)
    data["dc_antecedent_30d"] = rng.uniform(200, 600, n)
    data["fwi_event_day"] = rng.uniform(20, 80, n)

    # Burn probability correlated with wind + VPD
    logit = (
        -3.0
        + 0.15 * data["wind_speed_max_24h"]
        + 0.04 * data["vpd_event_day"]
        + 0.01 * data["fwi_event_day"]
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    burned = (rng.random(n) < prob).astype(int)

    data["burned"] = burned
    data["composite_score"] = rng.uniform(0, 1, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Feature list consistency
# ---------------------------------------------------------------------------

class TestFeatureLists:
    """Verify feature lists are correct and non-overlapping."""

    def test_structural_count(self):
        assert len(STRUCTURAL_FEATURES) == 21

    def test_dynamic_count(self):
        assert len(DYNAMIC_FEATURES) == 5

    def test_all_features_combined(self):
        assert len(ALL_FEATURES) == 26
        assert ALL_FEATURES == STRUCTURAL_FEATURES + DYNAMIC_FEATURES

    def test_no_duplicates(self):
        assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)

    def test_no_overlap(self):
        overlap = set(STRUCTURAL_FEATURES) & set(DYNAMIC_FEATURES)
        assert overlap == set(), f"Overlap: {overlap}"


# ---------------------------------------------------------------------------
# train_loeo
# ---------------------------------------------------------------------------

class TestTrainLoeo:
    """Tests for leave-one-event-out cross-validation."""

    @pytest.fixture
    def two_events(self):
        return {
            "event_a": _make_synthetic_event(n=400, seed=1),
            "event_b": _make_synthetic_event(n=300, burn_rate=0.15, seed=2),
        }

    def test_returns_all_events(self, two_events):
        results = train_loeo(two_events)
        assert set(results.keys()) == {"event_a", "event_b"}

    def test_prediction_shape(self, two_events):
        results = train_loeo(two_events)
        assert len(results["event_a"]["y_prob"]) == 400
        assert len(results["event_b"]["y_prob"]) == 300

    def test_probabilities_in_range(self, two_events):
        results = train_loeo(two_events)
        for res in results.values():
            assert np.all(res["y_prob"] >= 0.0)
            assert np.all(res["y_prob"] <= 1.0)

    def test_model_is_lgbm(self, two_events):
        import lightgbm as lgb

        results = train_loeo(two_events)
        for res in results.values():
            assert isinstance(res["model"], lgb.LGBMClassifier)

    def test_structural_only_features(self, two_events):
        """Can train with structural features only."""
        results = train_loeo(two_events, feature_cols=STRUCTURAL_FEATURES)
        for res in results.values():
            assert res["X_test"].shape[1] == 21

    def test_y_true_preserved(self, two_events):
        """y_true should match the original burned labels."""
        results = train_loeo(two_events)
        np.testing.assert_array_equal(
            results["event_a"]["y_true"],
            two_events["event_a"]["burned"].values,
        )


# ---------------------------------------------------------------------------
# train_full
# ---------------------------------------------------------------------------

class TestTrainFull:
    """Tests for full-data model training."""

    def test_returns_classifier(self):
        import lightgbm as lgb

        datasets = {
            "a": _make_synthetic_event(n=200, seed=10),
            "b": _make_synthetic_event(n=200, seed=20),
        }
        model = train_full(datasets)
        assert isinstance(model, lgb.LGBMClassifier)

    def test_uses_all_features(self):
        datasets = {
            "a": _make_synthetic_event(n=200, seed=10),
        }
        model = train_full(datasets)
        assert model.n_features_ == 26


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for AUC metric computation."""

    @pytest.fixture
    def loeo_and_datasets(self):
        datasets = {
            "event_a": _make_synthetic_event(n=500, seed=1),
            "event_b": _make_synthetic_event(n=400, seed=2),
        }
        loeo = train_loeo(datasets)
        return loeo, datasets

    def test_metrics_keys(self, loeo_and_datasets):
        loeo, datasets = loeo_and_datasets
        metrics = compute_metrics(loeo, datasets)
        assert set(metrics.keys()) == {"event_a", "event_b"}

    def test_metric_fields(self, loeo_and_datasets):
        loeo, datasets = loeo_and_datasets
        metrics = compute_metrics(loeo, datasets)
        for m in metrics.values():
            assert "auc_structural_v1" in m
            assert "auc_lgbm_v2" in m
            assert "auc_delta" in m
            assert "n_buildings" in m
            assert "n_burned" in m

    def test_auc_in_range(self, loeo_and_datasets):
        loeo, datasets = loeo_and_datasets
        metrics = compute_metrics(loeo, datasets)
        for m in metrics.values():
            assert 0.0 <= m["auc_structural_v1"] <= 1.0
            assert 0.0 <= m["auc_lgbm_v2"] <= 1.0

    def test_delta_consistency(self, loeo_and_datasets):
        loeo, datasets = loeo_and_datasets
        metrics = compute_metrics(loeo, datasets)
        for m in metrics.values():
            expected = m["auc_lgbm_v2"] - m["auc_structural_v1"]
            assert m["auc_delta"] == pytest.approx(expected, abs=1e-4)

    def test_population_counts(self, loeo_and_datasets):
        loeo, datasets = loeo_and_datasets
        metrics = compute_metrics(loeo, datasets)
        assert metrics["event_a"]["n_buildings"] == 500
        assert metrics["event_b"]["n_buildings"] == 400


# ---------------------------------------------------------------------------
# Leakage mitigation — fire history excluded from LOEO
# ---------------------------------------------------------------------------

class TestLeakageMitigation:
    """Verify fire history features are correctly excluded from LOEO."""

    def test_fire_history_count(self):
        assert len(FIRE_HISTORY_FEATURES) == 5

    def test_loeo_features_count(self):
        # 26 total - 5 fire history = 21 LOEO features
        assert len(LOEO_FEATURES) == 21

    def test_loeo_excludes_fire_history(self):
        assert set(FIRE_HISTORY_FEATURES).isdisjoint(set(LOEO_FEATURES))

    def test_loeo_features_subset_of_all(self):
        assert set(LOEO_FEATURES).issubset(set(ALL_FEATURES))

    def test_loeo_with_filtered_features(self):
        datasets = {
            "event_a": _make_synthetic_event(n=400, seed=1),
            "event_b": _make_synthetic_event(n=300, seed=2),
        }
        results = train_loeo(datasets, feature_cols=LOEO_FEATURES)
        for res in results.values():
            assert res["X_test"].shape[1] == 21
