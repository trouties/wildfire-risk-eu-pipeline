"""
tests/test_scoring_engine.py
-----------------------------
Unit tests for src/scoring/engine.py pure functions.

All tests use synthetic data only — no DuckDB, no network, no credentials.
Targets: percentile_rank_normalize, _is_eligible, _redistribute_weights,
         _classify_quintile, _score_group.
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring.engine import (
    _classify_quintile,
    _is_eligible,
    _redistribute_weights,
    _score_group,
    percentile_rank_normalize,
)

# ---------------------------------------------------------------------------
# percentile_rank_normalize
# ---------------------------------------------------------------------------

class TestPercentileRankNormalize:
    """Tests for the core normalization function."""

    def test_output_range_positive(self):
        """Scores should be in [0, 1] for positive direction."""
        s = pd.Series([10, 20, 30, 40, 50], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_monotonic_positive(self):
        """Higher values → higher scores for positive direction."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert list(result) == sorted(result), "positive direction should be monotonically increasing"

    def test_monotonic_negative(self):
        """Higher values → lower scores for negative direction (inverted)."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="feat")
        result = percentile_rank_normalize(s, direction="negative")
        assert list(result) == sorted(result, reverse=True), (
            "negative direction should be monotonically decreasing"
        )

    def test_inversion_relationship(self):
        """Positive + negative scores should sum to 1.0 for each element."""
        s = pd.Series([10, 20, 30, 40, 50], name="feat")
        pos = percentile_rank_normalize(s, direction="positive")
        neg = percentile_rank_normalize(s, direction="negative")
        np.testing.assert_allclose(pos + neg, 1.0)

    def test_neutral_treated_as_positive(self):
        """Neutral direction should behave identically to positive."""
        s = pd.Series([5, 15, 25, 35, 45], name="feat")
        pos = percentile_rank_normalize(s, direction="positive")
        neu = percentile_rank_normalize(s, direction="neutral")
        pd.testing.assert_series_equal(pos, neu)

    def test_ties_get_average_rank(self):
        """Tied values should get the same score."""
        s = pd.Series([1, 2, 2, 3], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert result.iloc[1] == result.iloc[2], "tied values must have equal scores"

    def test_single_element(self):
        """Single-element series should return 0.5."""
        s = pd.Series([42.0], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert result.iloc[0] == 0.5

    def test_preserves_index(self):
        """Output should have the same index as input."""
        idx = pd.Index([100, 200, 300])
        s = pd.Series([1, 2, 3], index=idx, name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        pd.testing.assert_index_equal(result.index, idx)

    def test_winsorization_clips_outliers(self):
        """Extreme outliers should be clipped to p1/p99."""
        # Create data with a big outlier
        vals = list(range(100)) + [10_000]
        s = pd.Series(vals, name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        # The outlier should not get a score much higher than its neighbors
        # after clipping — specifically, it should be == max (1.0)
        assert result.iloc[-1] == result.max()

    def test_constant_values(self):
        """All equal values should all get the same score (0.5 via average rank)."""
        s = pd.Series([7, 7, 7, 7], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert result.nunique() == 1, "constant input → all scores equal"
        assert result.iloc[0] == pytest.approx(0.5)

    def test_two_elements(self):
        """Two-element series: lower→0.0, higher→1.0 for positive."""
        s = pd.Series([1.0, 2.0], name="feat")
        result = percentile_rank_normalize(s, direction="positive")
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _is_eligible
# ---------------------------------------------------------------------------

class TestIsEligible:
    """Tests for the feature eligibility checker."""

    def test_normal_column_is_eligible(self):
        """A column with varied non-null values should be eligible."""
        s = pd.Series([1, 2, 3, 4, 5])
        eligible, reason = _is_eligible(s)
        assert eligible is True
        assert reason == ""

    def test_all_null_not_eligible(self):
        """An all-NULL column should be rejected."""
        s = pd.Series([None, None, None, None], dtype=float)
        eligible, reason = _is_eligible(s)
        assert eligible is False
        assert "all-NULL" in reason

    def test_high_null_fraction(self):
        """More than 50% NULL → not eligible."""
        s = pd.Series([1.0, None, None, None])
        eligible, reason = _is_eligible(s)
        assert eligible is False
        assert "high-NULL" in reason

    def test_exactly_50pct_null_is_eligible(self):
        """Exactly 50% NULL → eligible (threshold is >50%, not >=50%)."""
        s = pd.Series([1.0, 2.0, None, None])
        eligible, reason = _is_eligible(s)
        assert eligible is True

    def test_zero_variance_not_eligible(self):
        """A constant column (single unique non-null value) → zero-variance."""
        s = pd.Series([5, 5, 5, 5])
        eligible, reason = _is_eligible(s)
        assert eligible is False
        assert "zero-variance" in reason

    def test_zero_variance_with_nulls(self):
        """Single unique value + some NULLs → zero-variance."""
        s = pd.Series([3.0, 3.0, None])
        eligible, reason = _is_eligible(s)
        assert eligible is False
        assert "zero-variance" in reason

    def test_two_unique_values(self):
        """Two distinct values → eligible (variance > 0)."""
        s = pd.Series([0, 1, 0, 1])
        eligible, reason = _is_eligible(s)
        assert eligible is True


# ---------------------------------------------------------------------------
# _redistribute_weights
# ---------------------------------------------------------------------------

class TestRedistributeWeights:
    """Tests for the proportional weight redistribution."""

    def test_all_groups_present(self):
        """When all groups present, weights should be unchanged."""
        cfg = {"a": 0.4, "b": 0.3, "c": 0.3}
        result = _redistribute_weights(cfg, ["a", "b", "c"])
        assert result == pytest.approx(cfg)

    def test_one_group_missing(self):
        """Missing group's weight redistributed proportionally."""
        cfg = {"terrain": 0.20, "vegetation": 0.30,
               "fire_weather": 0.20, "fire_history": 0.20,
               "proximity": 0.10}
        present = ["terrain", "vegetation", "fire_weather", "fire_history"]
        result = _redistribute_weights(cfg, present)

        # Weights should sum to 1.0
        assert sum(result.values()) == pytest.approx(1.0)
        # Only present groups in result
        assert set(result.keys()) == set(present)
        # Vegetation should have highest weight (0.30 / 0.90)
        assert result["vegetation"] == pytest.approx(0.30 / 0.90)

    def test_weights_sum_to_one(self):
        """Redistributed weights must always sum to 1.0."""
        cfg = {"a": 0.5, "b": 0.3, "c": 0.2}
        for present in [["a"], ["a", "b"], ["a", "b", "c"]]:
            result = _redistribute_weights(cfg, present)
            assert sum(result.values()) == pytest.approx(1.0)

    def test_single_group_gets_full_weight(self):
        """If only one group present, it gets weight 1.0."""
        cfg = {"a": 0.5, "b": 0.3, "c": 0.2}
        result = _redistribute_weights(cfg, ["b"])
        assert result["b"] == pytest.approx(1.0)

    def test_preserves_relative_proportions(self):
        """Ratio between two present groups should be preserved."""
        cfg = {"a": 0.4, "b": 0.2, "c": 0.4}
        result = _redistribute_weights(cfg, ["a", "b"])
        # a:b ratio should be 0.4:0.2 = 2:1
        assert result["a"] / result["b"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _classify_quintile
# ---------------------------------------------------------------------------

class TestClassifyQuintile:
    """Tests for quintile classification."""

    def test_five_classes_produced(self):
        """Result should contain classes 1 through 5."""
        scores = pd.Series(np.linspace(0, 1, 1000))
        result = _classify_quintile(scores)
        assert set(result.unique()) == {1, 2, 3, 4, 5}

    def test_roughly_equal_bins(self):
        """Each class should contain ~20% of the population."""
        n = 10_000
        scores = pd.Series(np.random.default_rng(42).random(n))
        result = _classify_quintile(scores)
        counts = result.value_counts()
        for cls in [1, 2, 3, 4, 5]:
            frac = counts[cls] / n
            assert 0.18 <= frac <= 0.22, f"Class {cls}: {frac:.2%} outside 18-22%"

    def test_ordering(self):
        """Higher scores should generally map to higher classes."""
        scores = pd.Series(np.linspace(0, 1, 100))
        result = _classify_quintile(scores)
        # Lowest score should be class 1, highest should be class 5
        assert result.iloc[0] == 1
        assert result.iloc[-1] == 5

    def test_integer_output(self):
        """Classes should be integers."""
        scores = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
        result = _classify_quintile(scores)
        assert result.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# _score_group
# ---------------------------------------------------------------------------

class TestScoreGroup:
    """Tests for the group scoring function."""

    @pytest.fixture
    def terrain_df(self):
        """Synthetic DataFrame with terrain columns."""
        rng = np.random.default_rng(123)
        n = 200
        return pd.DataFrame({
            "elevation_m":        rng.uniform(0, 800, n),
            "slope_deg":          rng.uniform(0, 45, n),
            "south_aspect_score": rng.uniform(0, 1, n),
            "tpi_300m":           rng.uniform(-5, 5, n),
            "tri_300m":           rng.uniform(0, 30, n),
            # aspect_deg is in ALWAYS_EXCLUDE
            "aspect_deg":         rng.uniform(0, 360, n),
        })

    def test_group_score_range(self, terrain_df):
        """Group score should be in [0, 1]."""
        gscore, _ = _score_group(terrain_df, "terrain")
        assert gscore.min() >= 0.0
        assert gscore.max() <= 1.0

    def test_excluded_features_not_in_group(self, terrain_df):
        """aspect_deg is not in _GROUP_FEATURES['terrain'], so it should not
        appear in the returned feature_scores dict at all."""
        _, feat_scores = _score_group(terrain_df, "terrain")
        assert "aspect_deg" not in feat_scores

    def test_eligible_features_have_scores(self, terrain_df):
        """Non-excluded features with variance should have score Series."""
        _, feat_scores = _score_group(terrain_df, "terrain")
        for col in ["elevation_m", "slope_deg", "south_aspect_score", "tpi_300m", "tri_300m"]:
            assert feat_scores[col] is not None, f"{col} should have a score"
            assert len(feat_scores[col]) == len(terrain_df)

    def test_missing_column_skipped(self):
        """If a group column is missing from the DataFrame, it should be skipped."""
        df = pd.DataFrame({
            "elevation_m": [1, 2, 3],
            "slope_deg":   [10, 20, 30],
            # Missing: south_aspect_score, tpi_300m, tri_300m, aspect_deg
        })
        gscore, feat_scores = _score_group(df, "terrain")
        assert feat_scores.get("south_aspect_score") is None
        assert gscore is not None

    def test_raises_on_no_eligible_features(self):
        """Should raise ValueError if no features are eligible."""
        # All terrain columns either excluded or missing
        df = pd.DataFrame({
            "aspect_deg": [1, 2, 3],  # in ALWAYS_EXCLUDE
        })
        with pytest.raises(ValueError, match="No eligible features"):
            _score_group(df, "terrain")

    @pytest.fixture
    def vegetation_df(self):
        """Synthetic DataFrame with vegetation columns."""
        rng = np.random.default_rng(456)
        n = 300
        return pd.DataFrame({
            "veg_fraction_100m":   rng.uniform(0, 1, n),
            "veg_fraction_500m":   rng.uniform(0, 1, n),
            "dist_to_forest_m":    rng.exponential(500, n),
            "dist_to_scrubland_m": rng.exponential(300, n),
            "wui_class":           rng.choice([0, 1, 2, 3], n),
            "veg_continuity_500m": rng.uniform(0, 1, n),
            # ndvi columns are in ALWAYS_EXCLUDE
            "ndvi_mean_100m":      [None] * n,
            "ndvi_mean_500m":      [None] * n,
            "ndvi_max_500m":       [None] * n,
        })

    def test_vegetation_group(self, vegetation_df):
        """Vegetation group should score all eligible features."""
        gscore, feat_scores = _score_group(vegetation_df, "vegetation")
        assert gscore.min() >= 0.0
        assert gscore.max() <= 1.0
        # NDVI features should be None (ALWAYS_EXCLUDE)
        assert feat_scores.get("ndvi_mean_100m") is None
        assert feat_scores.get("ndvi_mean_500m") is None
        assert feat_scores.get("ndvi_max_500m") is None

    def test_negative_direction_inversion(self, vegetation_df):
        """dist_to_forest_m (negative direction) should invert scores."""
        _, feat_scores = _score_group(vegetation_df, "vegetation")
        forest_scores = feat_scores["dist_to_forest_m"].dropna()
        veg_frac_scores = feat_scores["veg_fraction_100m"].dropna()
        # Both should be in [0, 1]
        assert forest_scores.min() >= 0.0
        assert forest_scores.max() <= 1.0
        assert veg_frac_scores.min() >= 0.0
        assert veg_frac_scores.max() <= 1.0
