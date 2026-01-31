"""
tests/test_dynamic_event.py
-----------------------------
Unit tests for src/features/dynamic_event.py.

All tests use synthetic data only — no CDS downloads, no real NetCDF, no DuckDB.
Targets: vpd_hpa, circular_std, compute_dynamic_features, assign_to_buildings.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr  # noqa: F401 (used by _make_*_nc helpers)

from src.features.dynamic_event import (
    FEATURE_COLS,
    assign_to_buildings,
    circular_std,
    compute_dynamic_features,
    vpd_hpa,
)

# ---------------------------------------------------------------------------
# vpd_hpa
# ---------------------------------------------------------------------------

class TestVpdHpa:
    """Tests for the vapor pressure deficit computation."""

    def test_zero_vpd_when_saturated(self):
        """VPD should be 0 when T == Td (100% RH)."""
        t_k = np.array([300.0, 290.0, 280.0])
        td_k = t_k.copy()
        vpd = vpd_hpa(t_k, td_k)
        np.testing.assert_allclose(vpd, 0.0, atol=1e-10)

    def test_positive_vpd(self):
        """VPD should be positive when T > Td."""
        t_k = np.array([310.0])   # 36.85 °C
        td_k = np.array([290.0])  # 16.85 °C
        vpd = vpd_hpa(t_k, td_k)
        assert vpd[0] > 0.0, "VPD should be positive when T > Td"

    def test_vpd_increases_with_temperature(self):
        """Higher T with same Td → higher VPD."""
        td_k = np.array([285.0, 285.0, 285.0])  # constant dewpoint
        t_k = np.array([290.0, 300.0, 310.0])    # increasing temperature
        vpd = vpd_hpa(t_k, td_k)
        assert vpd[0] < vpd[1] < vpd[2], "VPD should increase with temperature"

    def test_vpd_never_negative(self):
        """VPD should be clipped to >= 0 even if Td > T (numerical edge)."""
        t_k = np.array([280.0])
        td_k = np.array([281.0])  # dewpoint above temp
        vpd = vpd_hpa(t_k, td_k)
        assert vpd[0] >= 0.0

    def test_known_value(self):
        """Verify against a hand-calculated value.

        T = 35°C (308.15 K):  es = 6.112 * exp(17.67*35/278.5) ≈ 56.24 hPa
        Td = 15°C (288.15 K): ea = 6.112 * exp(17.67*15/258.5) ≈ 17.05 hPa
        VPD ≈ 39.2 hPa
        """
        t_k = np.array([308.15])
        td_k = np.array([288.15])
        vpd = vpd_hpa(t_k, td_k)
        assert 38.0 < vpd[0] < 41.0, f"Expected ~39.2 hPa, got {vpd[0]:.1f}"


# ---------------------------------------------------------------------------
# circular_std
# ---------------------------------------------------------------------------

class TestCircularStd:
    """Tests for the circular standard deviation computation."""

    def test_constant_direction(self):
        """Circular std ≈ 0 for identical angles."""
        angles = np.full((20, 3, 3), np.pi / 4)  # all 45°
        std = circular_std(angles, axis=0)
        np.testing.assert_allclose(std, 0.0, atol=0.01)

    def test_uniform_distribution(self):
        """Circular std should be maximal for uniformly distributed angles."""
        # Angles uniformly spaced around circle
        n = 1000
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).reshape(-1, 1, 1)
        angles = np.broadcast_to(angles, (n, 2, 2)).copy()
        std = circular_std(angles, axis=0)
        # For uniform distribution, R → 0, so std → sqrt(-2*ln(ε)) ≈ large
        assert np.all(std > 2.0), "Uniform angles should have high circular std"

    def test_opposite_directions(self):
        """Two opposite directions should have high std."""
        angles = np.array([0.0, np.pi, 0.0, np.pi]).reshape(4, 1)
        std = circular_std(angles, axis=0)
        assert std[0] > 1.0, "Opposite directions should yield high std"

    def test_shape_preserved(self):
        """Output shape should be input shape minus the axis dimension."""
        angles = np.random.default_rng(42).uniform(-np.pi, np.pi, (24, 5, 7))
        std = circular_std(angles, axis=0)
        assert std.shape == (5, 7)


# ---------------------------------------------------------------------------
# compute_dynamic_features — synthetic NetCDF integration
# ---------------------------------------------------------------------------

def _make_hourly_nc(path, lats, lons, times, u10, v10, t2m, d2m):
    """Write a synthetic hourly ERA5 NetCDF."""
    ds = xr.Dataset(
        {
            "u10": (["valid_time", "latitude", "longitude"], u10),
            "v10": (["valid_time", "latitude", "longitude"], v10),
            "t2m": (["valid_time", "latitude", "longitude"], t2m),
            "d2m": (["valid_time", "latitude", "longitude"], d2m),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path)
    return path


def _make_fwi_nc(path, lats, lons, times, dc, fwi):
    """Write a synthetic daily FWI NetCDF."""
    ds = xr.Dataset(
        {
            "dc": (["time", "latitude", "longitude"], dc),
            "fwi": (["time", "latitude", "longitude"], fwi),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(path)
    return path


class TestComputeDynamicFeatures:
    """Integration tests for compute_dynamic_features with synthetic data."""

    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create minimal synthetic hourly + FWI NetCDF files."""
        lats = np.array([38.0, 38.1, 38.2])
        lons = np.array([23.5, 23.6, 23.7])
        event_date = "2021-08-03"

        # 48 hourly timesteps: Aug 2 00:00 → Aug 3 23:00
        hourly_times = pd.date_range("2021-08-02", periods=48, freq="h")
        n_t, n_lat, n_lon = 48, 3, 3

        # Constant wind from north: u=0, v=-10 m/s (steady)
        u10 = np.zeros((n_t, n_lat, n_lon))
        v10 = np.full((n_t, n_lat, n_lon), -10.0)

        # One spike: hour 30 (Aug 3, 06:00) has strong NW gust
        u10[30] = -15.0
        v10[30] = -15.0  # ws = sqrt(225+225) ≈ 21.2 m/s

        # Temperature: ~35°C (308.15 K), dewpoint: ~15°C (288.15 K)
        t2m = np.full((n_t, n_lat, n_lon), 308.15)
        d2m = np.full((n_t, n_lat, n_lon), 288.15)

        hourly_nc = _make_hourly_nc(
            tmp_path / "test_hourly.nc", lats, lons, hourly_times,
            u10, v10, t2m, d2m,
        )

        # FWI: 60 daily timesteps (Jul 1 → Aug 29)
        fwi_times = pd.date_range("2021-07-01", periods=60, freq="D")
        n_fwi_t = 60
        dc_vals = np.full((n_fwi_t, n_lat, n_lon), 400.0)
        fwi_vals = np.full((n_fwi_t, n_lat, n_lon), 50.0)
        # Event day (Aug 3 = index 33) has high FWI
        fwi_vals[33] = 66.4

        fwi_nc = _make_fwi_nc(
            tmp_path / "test_fwi.nc", lats, lons, fwi_times,
            dc_vals, fwi_vals,
        )

        return hourly_nc, fwi_nc, event_date

    def test_output_shape(self, synthetic_data):
        """Should return one row per grid cell (3×3 = 9)."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        assert len(df) == 9  # 3 lat × 3 lon
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_wind_speed_max_captures_gust(self, synthetic_data):
        """Wind max should capture the NW gust at hour 30."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        expected_max = np.sqrt(15.0**2 + 15.0**2)  # ≈ 21.2 m/s
        np.testing.assert_allclose(
            df["wind_speed_max_24h"].max(), expected_max, rtol=0.01,
        )

    def test_wind_dir_consistency_high_for_steady(self, synthetic_data):
        """Mostly steady northerly wind → high consistency."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        # With 11 of 12 hours having identical direction, consistency should be high
        assert df["wind_dir_consistency"].mean() > 1.0

    def test_vpd_positive(self, synthetic_data):
        """VPD should be positive (T > Td in synthetic data)."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        assert (df["vpd_event_day"] > 0).all()

    def test_dc_antecedent_value(self, synthetic_data):
        """DC antecedent should match synthetic constant (400.0)."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        np.testing.assert_allclose(df["dc_antecedent_30d"].values, 400.0)

    def test_fwi_event_day_value(self, synthetic_data):
        """FWI event day should match the Aug 3 spike (66.4)."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        np.testing.assert_allclose(df["fwi_event_day"].values, 66.4, rtol=0.01)

    def test_no_nans(self, synthetic_data):
        """No NaN values in features for land cells."""
        hourly_nc, fwi_nc, event_date = synthetic_data
        df = compute_dynamic_features(hourly_nc, fwi_nc, event_date)
        assert not df[FEATURE_COLS].isna().any().any()


# ---------------------------------------------------------------------------
# assign_to_buildings
# ---------------------------------------------------------------------------

class TestAssignToBuildings:
    """Tests for cKDTree building assignment."""

    def test_correct_assignment(self):
        """Buildings should be assigned to nearest grid cell."""
        grid = pd.DataFrame({
            "latitude":  [38.0, 38.0, 38.1, 38.1],
            "longitude": [23.5, 23.6, 23.5, 23.6],
            "wind_speed_max_24h":  [10.0, 20.0, 30.0, 40.0],
            "wind_dir_consistency": [1.0, 2.0, 3.0, 4.0],
            "vpd_event_day":       [5.0, 10.0, 15.0, 20.0],
            "dc_antecedent_30d":   [100.0, 200.0, 300.0, 400.0],
            "fwi_event_day":       [50.0, 60.0, 70.0, 80.0],
        })

        buildings = pd.DataFrame({
            "building_id": ["b1", "b2"],
            "centroid_lat": [38.01, 38.09],  # b1 near (38.0,23.5), b2 near (38.1,23.5)
            "centroid_lon": [23.51, 23.51],
        })

        result = assign_to_buildings(grid, buildings, event_id="test")
        assert result.loc[0, "wind_speed_max_24h"] == 10.0  # nearest to cell 0
        assert result.loc[1, "wind_speed_max_24h"] == 30.0  # nearest to cell 2
        assert (result["event_id"] == "test").all()

    def test_sea_cells_excluded(self):
        """Grid cells with NaN features (sea) should be skipped."""
        grid = pd.DataFrame({
            "latitude":  [38.0, 38.1],
            "longitude": [23.5, 23.5],
            "wind_speed_max_24h":  [np.nan, 20.0],
            "wind_dir_consistency": [np.nan, 2.0],
            "vpd_event_day":       [np.nan, 10.0],
            "dc_antecedent_30d":   [np.nan, 200.0],
            "fwi_event_day":       [np.nan, 60.0],
        })

        buildings = pd.DataFrame({
            "building_id": ["b1"],
            "centroid_lat": [38.0],  # closest to sea cell, but should get land cell
            "centroid_lon": [23.5],
        })

        result = assign_to_buildings(grid, buildings, event_id="test")
        assert result.loc[0, "wind_speed_max_24h"] == 20.0  # assigned to land cell

    def test_all_nan_raises(self):
        """Should raise if no valid land cells exist."""
        grid = pd.DataFrame({
            "latitude":  [38.0],
            "longitude": [23.5],
            "wind_speed_max_24h":  [np.nan],
            "wind_dir_consistency": [np.nan],
            "vpd_event_day":       [np.nan],
            "dc_antecedent_30d":   [np.nan],
            "fwi_event_day":       [np.nan],
        })

        buildings = pd.DataFrame({
            "building_id": ["b1"],
            "centroid_lat": [38.0],
            "centroid_lon": [23.5],
        })

        with pytest.raises(ValueError, match="No valid land cells"):
            assign_to_buildings(grid, buildings, event_id="test")

    def test_output_columns(self):
        """Result should have event_id, building_id, and all 5 feature columns."""
        grid = pd.DataFrame({
            "latitude":  [38.0],
            "longitude": [23.5],
            "wind_speed_max_24h":  [10.0],
            "wind_dir_consistency": [1.0],
            "vpd_event_day":       [5.0],
            "dc_antecedent_30d":   [100.0],
            "fwi_event_day":       [50.0],
        })

        buildings = pd.DataFrame({
            "building_id": ["b1", "b2", "b3"],
            "centroid_lat": [38.0, 38.0, 38.0],
            "centroid_lon": [23.5, 23.5, 23.5],
        })

        result = assign_to_buildings(grid, buildings, event_id="ev1")
        expected_cols = {"event_id", "building_id"} | set(FEATURE_COLS)
        assert set(result.columns) == expected_cols
        assert len(result) == 3
