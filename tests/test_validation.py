"""
tests/test_validation.py
-------------------------
Unit tests for src/validation/validator.py pure functions.

All tests use synthetic data only — no DuckDB, no network, no credentials.
Targets: _lift_at_topk, compute_metrics, failure_analysis.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.validator import (
    _lift_at_topk,
    compute_metrics,
    failure_analysis,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic validation DataFrames
# ---------------------------------------------------------------------------

def _make_validation_df(
    n: int = 500,
    n_burned: int = 50,
    seed: int = 42,
    discriminative: bool = True,
) -> pd.DataFrame:
    """Create a synthetic validation DataFrame with required columns.

    Args:
        n:              Total buildings.
        n_burned:       Number of burned buildings.
        seed:           RNG seed for reproducibility.
        discriminative: If True, burned buildings get higher scores on average
                        (model has skill). If False, scores are random.

    Returns:
        DataFrame with columns matching compute_metrics expectations.
    """
    rng = np.random.default_rng(seed)

    burned = np.zeros(n, dtype=int)
    burned[:n_burned] = 1
    rng.shuffle(burned)

    if discriminative:
        # Burned buildings get higher composite scores on average
        composite = np.where(
            burned == 1,
            rng.beta(5, 2, n),   # mean ~0.71
            rng.beta(2, 5, n),   # mean ~0.29
        )
    else:
        composite = rng.uniform(0, 1, n)

    # Generate sub-scores
    score_terrain = rng.uniform(0, 1, n)
    score_vegetation = rng.uniform(0, 1, n)
    score_fire_weather = rng.uniform(0, 1, n)
    score_fire_history = rng.uniform(0, 1, n)

    # Baseline score
    baseline = rng.uniform(0, 1, n)

    # Risk class from composite quintiles
    risk_class = pd.qcut(composite, q=5, labels=[1, 2, 3, 4, 5],
                         duplicates="drop").astype(int)

    return pd.DataFrame({
        "building_id":        [f"B{i:06d}" for i in range(n)],
        "burned":             burned,
        "composite_score":    composite,
        "baseline_score":     baseline,
        "risk_class":         risk_class,
        "score_terrain":      score_terrain,
        "score_vegetation":   score_vegetation,
        "score_fire_weather": score_fire_weather,
        "score_fire_history": score_fire_history,
        "dist_to_forest_m":   rng.exponential(500, n),
    })


# ---------------------------------------------------------------------------
# _lift_at_topk
# ---------------------------------------------------------------------------

class TestLiftAtTopK:
    """Tests for the lift metric calculation."""

    def test_perfect_model(self):
        """A perfect model should achieve maximum lift at top-k."""
        # 10 positives, 90 negatives; perfect model ranks all positives first
        y_true = np.array([1]*10 + [0]*90)
        y_score = np.array([1.0]*10 + [0.0]*90)
        lift = _lift_at_topk(y_true, y_score, k=0.10)
        # top 10% = 10 buildings, all positive → precision=1.0, prevalence=0.1
        assert lift == pytest.approx(10.0)

    def test_random_model_lift_near_one(self):
        """A random model should have lift ≈ 1.0 (no better than chance)."""
        rng = np.random.default_rng(99)
        n = 10_000
        y_true = np.zeros(n, dtype=int)
        y_true[:1000] = 1
        rng.shuffle(y_true)
        y_score = rng.uniform(0, 1, n)
        lift = _lift_at_topk(y_true, y_score, k=0.10)
        assert 0.7 <= lift <= 1.3, f"Random model lift {lift} too far from 1.0"

    def test_zero_prevalence_returns_nan(self):
        """If no positives exist, lift should be NaN."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        lift = _lift_at_topk(y_true, y_score, k=0.20)
        assert np.isnan(lift)

    def test_k_parameter(self):
        """Different k values should return different lifts."""
        y_true = np.array([1]*10 + [0]*90)
        y_score = np.array(list(range(100, 0, -1)), dtype=float)
        # positives have the highest scores
        lift_10 = _lift_at_topk(y_true, y_score, k=0.10)
        lift_50 = _lift_at_topk(y_true, y_score, k=0.50)
        assert lift_10 >= lift_50, "Lift at smaller k should be >= lift at larger k for good model"

    def test_minimum_one_building(self):
        """Even with tiny k, at least 1 building should be examined."""
        y_true = np.array([1, 0, 0])
        y_score = np.array([0.9, 0.1, 0.2])
        # k=0.01 on n=3 → ceil(0.03) = 1 building
        lift = _lift_at_topk(y_true, y_score, k=0.01)
        assert lift > 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for the full metrics computation."""

    def test_discriminative_model_auc_high(self):
        """A discriminative model should achieve AUC > 0.70."""
        df = _make_validation_df(n=1000, n_burned=100, discriminative=True)
        metrics = compute_metrics(df)
        assert metrics["model"]["auc_roc"] > 0.70

    def test_random_model_auc_near_half(self):
        """A non-discriminative model should have AUC ≈ 0.50."""
        df = _make_validation_df(n=2000, n_burned=200, discriminative=False, seed=7)
        metrics = compute_metrics(df)
        auc = metrics["model"]["auc_roc"]
        assert 0.40 <= auc <= 0.60, f"Random model AUC {auc} should be near 0.50"

    def test_metrics_structure(self):
        """Returned dict should have all expected top-level keys."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        expected_keys = {
            "event", "event_date", "validation_bbox", "population",
            "model", "baseline", "model_vs_baseline", "class_stats",
            "mean_scores", "geographic_diagnostic", "thresholds",
        }
        assert expected_keys == set(metrics.keys())

    def test_population_counts(self):
        """Population stats should match input data."""
        n, n_burned = 500, 50
        df = _make_validation_df(n=n, n_burned=n_burned)
        metrics = compute_metrics(df)
        assert metrics["population"]["n_buildings"] == n
        assert metrics["population"]["n_burned"] == n_burned
        assert metrics["population"]["n_unburned"] == n - n_burned
        assert metrics["population"]["prevalence"] == pytest.approx(n_burned / n)

    def test_model_metrics_in_range(self):
        """Model metrics should be in valid ranges."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        m = metrics["model"]
        assert 0.0 <= m["auc_roc"] <= 1.0
        assert m["lift_at_top10pct"] >= 0.0
        assert 0.0 <= m["precision_cls5"] <= 1.0
        assert 0.0 <= m["recall_cls45"] <= 1.0

    def test_baseline_metrics_present(self):
        """Baseline comparison metrics should be present."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        assert "auc_roc" in metrics["baseline"]
        assert "lift_at_top10pct" in metrics["baseline"]
        assert "auc_roc_delta" in metrics["model_vs_baseline"]

    def test_class_stats_all_five_classes(self):
        """Class stats should cover classes 1–5."""
        df = _make_validation_df(n=1000, n_burned=100)
        metrics = compute_metrics(df)
        for cls in ["1", "2", "3", "4", "5"]:
            assert cls in metrics["class_stats"]
            stat = metrics["class_stats"][cls]
            assert "n_buildings" in stat
            assert "n_burned" in stat
            assert "burned_rate" in stat

    def test_mean_scores_burned_vs_unburned(self):
        """Mean scores dict should contain burned and unburned entries."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        for group in ["burned", "unburned"]:
            assert group in metrics["mean_scores"]
            for col in ["composite_score", "score_terrain", "score_vegetation",
                        "score_fire_weather", "score_fire_history"]:
                assert col in metrics["mean_scores"][group]

    def test_discriminative_burned_higher_composite(self):
        """For a discriminative model, burned should have higher mean composite."""
        df = _make_validation_df(n=2000, n_burned=200, discriminative=True)
        metrics = compute_metrics(df)
        burned_mean = metrics["mean_scores"]["burned"]["composite_score"]
        unburned_mean = metrics["mean_scores"]["unburned"]["composite_score"]
        assert burned_mean > unburned_mean

    def test_geographic_diagnostic_without_lat(self):
        """Without centroid_lat column, geographic_diagnostic should be empty."""
        df = _make_validation_df()
        assert "centroid_lat" not in df.columns
        metrics = compute_metrics(df)
        assert metrics["geographic_diagnostic"] == {}

    def test_geographic_diagnostic_with_lat(self):
        """With centroid_lat column, geographic_diagnostic should have zones."""
        df = _make_validation_df(n=1000, n_burned=100)
        # Add lat column spanning the split threshold (38.05)
        rng = np.random.default_rng(11)
        df["centroid_lat"] = rng.uniform(37.98, 38.12, len(df))
        metrics = compute_metrics(df)
        geo = metrics["geographic_diagnostic"]
        assert "south" in geo
        assert "north" in geo
        for zone in ["south", "north"]:
            assert "n_buildings" in geo[zone]
            assert "auc_roc_model" in geo[zone]

    def test_thresholds_present(self):
        """Thresholds for pass/fail should be present."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        assert metrics["thresholds"]["auc_failure"] == 0.60
        assert metrics["thresholds"]["auc_target"] == 0.70

    def test_delta_consistency(self):
        """model_vs_baseline deltas should equal model - baseline."""
        df = _make_validation_df()
        metrics = compute_metrics(df)
        auc_delta = metrics["model"]["auc_roc"] - metrics["baseline"]["auc_roc"]
        assert metrics["model_vs_baseline"]["auc_roc_delta"] == pytest.approx(auc_delta)
        lift_delta = metrics["model"]["lift_at_top10pct"] - metrics["baseline"]["lift_at_top10pct"]
        assert metrics["model_vs_baseline"]["lift_delta"] == pytest.approx(lift_delta)


# ---------------------------------------------------------------------------
# failure_analysis
# ---------------------------------------------------------------------------

class TestFailureAnalysis:
    """Tests for the failure analysis function."""

    @pytest.fixture
    def validation_df(self):
        return _make_validation_df(n=500, n_burned=50, seed=42)

    def test_returns_both_keys(self, validation_df):
        """Result should have false_negatives and false_positives."""
        result = failure_analysis(validation_df)
        assert "false_negatives" in result
        assert "false_positives" in result

    def test_false_negatives_are_burned(self, validation_df):
        """False negatives should all be burned buildings."""
        result = failure_analysis(validation_df)
        fn = result["false_negatives"]
        assert (fn["burned"] == 1).all()

    def test_false_positives_are_unburned(self, validation_df):
        """False positives should all be unburned buildings."""
        result = failure_analysis(validation_df)
        fp = result["false_positives"]
        assert (fp["burned"] == 0).all()

    def test_false_negatives_sorted_ascending(self, validation_df):
        """False negatives should be sorted by composite_score ascending (lowest first)."""
        result = failure_analysis(validation_df)
        fn = result["false_negatives"]
        scores = fn["composite_score"].values
        assert list(scores) == sorted(scores)

    def test_false_positives_sorted_descending(self, validation_df):
        """False positives should be sorted by composite_score descending (highest first)."""
        result = failure_analysis(validation_df)
        fp = result["false_positives"]
        scores = fp["composite_score"].values
        assert list(scores) == sorted(scores, reverse=True)

    def test_n_top_parameter(self, validation_df):
        """Custom n_top should limit the number of rows returned."""
        result = failure_analysis(validation_df, n_top=5)
        assert len(result["false_negatives"]) <= 5
        assert len(result["false_positives"]) <= 5

    def test_default_n_top_is_10(self, validation_df):
        """Default should return up to 10 rows each."""
        result = failure_analysis(validation_df)
        assert len(result["false_negatives"]) <= 10
        assert len(result["false_positives"]) <= 10

    def test_required_columns_present(self, validation_df):
        """Output DataFrames should contain expected columns."""
        result = failure_analysis(validation_df)
        expected_cols = {
            "building_id", "burned", "composite_score", "risk_class",
            "score_terrain", "score_vegetation", "score_fire_weather",
            "score_fire_history", "baseline_score", "dist_to_forest_m",
        }
        for key in ["false_negatives", "false_positives"]:
            assert set(result[key].columns) == expected_cols

    def test_no_burned_buildings(self):
        """If there are no burned buildings, false_negatives should be empty."""
        df = _make_validation_df(n=100, n_burned=0)
        # Fix: with 0 burned, y_true is all zeros. Make sure compute is safe.
        result = failure_analysis(df, n_top=5)
        assert len(result["false_negatives"]) == 0
        assert len(result["false_positives"]) == 5

    def test_all_burned_buildings(self):
        """If all buildings burned, false_positives should be empty."""
        df = _make_validation_df(n=100, n_burned=100)
        result = failure_analysis(df, n_top=5)
        assert len(result["false_positives"]) == 0
        assert len(result["false_negatives"]) == 5
