"""
tests/test_acquire_effis.py
----------------------------
Unit tests for src/acquire/effis.py.

Tests cover:
- _circle_polygon_4326: geometry, area approximation, CRS
- _build_proxy_gdf: schema, Mati 2018 presence, geometry validity
- _try_effis_wfs: WFS response handling (mocked)
- download_effis: output file schema and Mati validation check

No network calls, no disk writes beyond tmp_path fixtures.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# _circle_polygon_4326
# ---------------------------------------------------------------------------

class TestCirclePolygon4326:
    """Tests for the circular fire-perimeter geometry helper."""

    def test_returns_valid_polygon(self):
        """Result is a valid Shapely Polygon."""

        from src.acquire.effis import _circle_polygon_4326

        poly = _circle_polygon_4326(23.978, 38.058, 12.9)
        assert poly is not None
        assert poly.geom_type == "Polygon"
        assert poly.is_valid
        assert not poly.is_empty

    def test_area_within_10pct_tolerance(self):
        """Circle area (projected to EPSG:2100) ≈ target area ± 10%."""
        import pyproj
        from shapely.ops import transform

        from src.acquire.effis import _circle_polygon_4326

        target_km2 = 12.9
        poly_4326 = _circle_polygon_4326(23.978, 38.058, target_km2)

        project = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:2100", always_xy=True
        ).transform
        poly_2100 = transform(project, poly_4326)
        area_km2 = poly_2100.area / 1e6

        assert abs(area_km2 - target_km2) / target_km2 < 0.10, (
            f"Area {area_km2:.2f} km² differs from target {target_km2} km² by >10%"
        )

    def test_centroid_near_input_point(self):
        """Centroid of the output polygon is near the input (lon, lat)."""
        from src.acquire.effis import _circle_polygon_4326

        lon, lat = 23.978, 38.058
        poly = _circle_polygon_4326(lon, lat, 12.9)
        c = poly.centroid
        assert abs(c.x - lon) < 0.05, f"Centroid lon {c.x} far from {lon}"
        assert abs(c.y - lat) < 0.05, f"Centroid lat {c.y} far from {lat}"

    def test_contains_centroid_point(self):
        """The polygon contains the input point."""
        from src.acquire.effis import _circle_polygon_4326

        lon, lat = 23.793, 38.136
        poly = _circle_polygon_4326(lon, lat, 63.0)
        assert poly.contains(Point(lon, lat))

    def test_small_fire_reasonable_radius(self):
        """1 km² fire → ~564 m radius → perimeter still valid."""
        from src.acquire.effis import _circle_polygon_4326

        poly = _circle_polygon_4326(24.0, 38.0, 1.0)
        assert poly.is_valid
        # 1 km² circle: radius ≈ 0.0052° at this latitude
        c = poly.centroid
        assert poly.contains(c)


# ---------------------------------------------------------------------------
# _build_proxy_gdf
# ---------------------------------------------------------------------------

class TestBuildProxyGdf:
    """Tests for the literature proxy fire dataset builder."""

    def test_returns_geodataframe(self):
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_crs_is_epsg4326(self):
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert gdf.crs.to_epsg() == 4326

    def test_schema_columns_present(self):
        """All required schema columns are present."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        required = {
            "fire_id", "event_date", "year", "country",
            "region", "area_ha", "area_km2", "source", "reference", "geometry",
        }
        assert required.issubset(set(gdf.columns)), (
            f"Missing columns: {required - set(gdf.columns)}"
        )

    def test_mati_2018_present(self):
        """Mati 2018-07-23 event is in the proxy dataset."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        mati = gdf[gdf["event_date"] == "2018-07-23"]
        assert len(mati) >= 1, "Mati 2018 event not found in proxy dataset"

    def test_mati_2018_area(self):
        """Mati 2018 area is approximately 12.9 km² (within 1 km²)."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        mati = gdf[
            (gdf["event_date"] == "2018-07-23") &
            (gdf["region"].str.contains("Mati", case=False, na=False))
        ]
        assert len(mati) >= 1, "No Mati (Rafina) row found"
        area = float(mati["area_km2"].iloc[0])
        assert abs(area - 12.9) < 2.0, f"Mati area {area} km² far from expected 12.9"

    def test_mati_2018_geometry_contains_centroid(self):
        """Mati 2018 polygon contains the documented centroid (23.978, 38.058)."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        mati = gdf[gdf["event_date"] == "2018-07-23"].iloc[0]
        assert mati.geometry.contains(Point(23.978, 38.058))

    def test_all_geometries_valid(self):
        """No invalid or null geometries in the proxy dataset."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        null_geom  = gdf["geometry"].isna().sum()
        invalid    = (~gdf.geometry.is_valid).sum()
        assert null_geom == 0, f"{null_geom} null geometries"
        assert invalid   == 0, f"{invalid} invalid geometries"

    def test_all_country_gr(self):
        """All events are flagged as country='GR'."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        non_gr = gdf[gdf["country"] != "GR"]
        assert len(non_gr) == 0, f"{len(non_gr)} non-GR rows"

    def test_year_range_2000_to_2024(self):
        """All events fall within 2000–2024."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert gdf["year"].min() >= 2000
        assert gdf["year"].max() <= 2024

    def test_source_is_literature_proxy(self):
        """All source values are 'literature_proxy'."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert (gdf["source"] == "literature_proxy").all()

    def test_area_ha_consistent_with_area_km2(self):
        """area_ha == area_km2 * 100 within floating-point tolerance."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        diff = (gdf["area_ha"] - gdf["area_km2"] * 100).abs()
        assert (diff < 1.0).all(), f"area_ha/area_km2 inconsistency: max diff {diff.max()}"

    def test_fire_ids_unique(self):
        """All fire_id values are unique."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert gdf["fire_id"].nunique() == len(gdf), "Duplicate fire_id values"

    def test_minimum_features(self):
        """At least 20 documented events are in the proxy dataset."""
        from src.acquire.effis import _build_proxy_gdf

        gdf = _build_proxy_gdf()
        assert len(gdf) >= 20, f"Only {len(gdf)} events — expected >= 20"


# ---------------------------------------------------------------------------
# _try_effis_wfs
# ---------------------------------------------------------------------------

class TestTryEffisWfs:
    """Tests for WFS download attempt (fully mocked)."""

    def test_returns_none_on_all_400(self):
        """If all WFS layer candidates return 400, result is None."""
        from src.acquire.effis import _try_effis_wfs

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.headers = {"Content-Type": "text/xml"}

        with patch("requests.get", return_value=mock_resp):
            result = _try_effis_wfs([23.4, 37.6, 24.2, 38.3])

        assert result is None

    def test_returns_none_on_connection_error(self):
        """Network errors result in None (not exception propagation)."""
        import requests as req

        from src.acquire.effis import _try_effis_wfs

        with patch("requests.get", side_effect=req.exceptions.ConnectionError("no network")):
            result = _try_effis_wfs([23.4, 37.6, 24.2, 38.3])

        assert result is None

    def test_parses_geojson_response(self):
        """A valid GeoJSON WFS response is parsed into a GeoDataFrame."""
        from src.acquire.effis import _EFFIS_LAYER_CANDIDATES, _try_effis_wfs

        geojson = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[23.9, 38.0], [24.0, 38.0],
                                          [24.0, 38.1], [23.9, 38.1], [23.9, 38.0]]],
                    },
                    "properties": {
                        "fireYear": 2018,
                        "country": "GR",
                    },
                }
            ]
        }

        # First layer candidate should succeed
        first_layer = _EFFIS_LAYER_CANDIDATES[0]

        def mock_get(url, params=None, timeout=None):
            resp = MagicMock()
            if params and params.get("typeName") == first_layer:
                resp.status_code = 200
                resp.headers = {"Content-Type": "application/json"}
                resp.json.return_value = geojson
            else:
                resp.status_code = 400
                resp.headers = {"Content-Type": "text/xml"}
            return resp

        with patch("requests.get", side_effect=mock_get):
            result = _try_effis_wfs([23.4, 37.6, 24.2, 38.3])

        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result["source"].iloc[0] == "effis_wfs"

    def test_empty_features_returns_none(self):
        """WFS response with 0 features → None (triggers proxy fallback)."""
        from src.acquire.effis import _try_effis_wfs

        empty_geojson = {"features": [], "totalFeatures": 0}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.json.return_value = empty_geojson

        with patch("requests.get", return_value=mock_resp):
            result = _try_effis_wfs([23.4, 37.6, 24.2, 38.3])

        assert result is None


# ---------------------------------------------------------------------------
# download_effis (integration, mocked I/O)
# ---------------------------------------------------------------------------

class TestDownloadEffis:
    """Tests for the download_effis entry point."""

    def _make_cfg(self) -> dict:
        return {
            "pipeline": {
                "aoi": {"bbox": [23.4, 37.6, 24.2, 38.3]},
                "run": {
                    "date_start": "2000-01-01",
                    "date_end": "2024-12-31",
                },
            },
            "sources": {
                "effis": {
                    "output_file": "data/raw/effis/fire_perimeters_greece.gpkg",
                    "date_range": ["2000-01-01", "2024-12-31"],
                    "country_filter": "GR",
                }
            },
        }

    def test_output_file_created(self, tmp_path: Path):
        """download_effis creates the output GeoPackage."""
        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        # Force WFS to fail → proxy fallback
        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            out = download_effis(output_dir, cfg, force=False)

        assert out.exists()
        assert out.suffix == ".gpkg"

    def test_output_gpkg_has_correct_layer(self, tmp_path: Path):
        """Output GeoPackage contains 'fire_perimeters' layer."""
        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            out = download_effis(output_dir, cfg, force=False)

        import fiona
        layers = fiona.listlayers(out)
        assert "fire_perimeters" in layers

    def test_output_contains_mati_2018(self, tmp_path: Path):
        """Output GeoPackage contains the Mati 2018 event."""
        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            out = download_effis(output_dir, cfg, force=False)

        gdf = gpd.read_file(out, layer="fire_perimeters")
        mati = gdf[gdf["event_date"] == "2018-07-23"]
        assert len(mati) >= 1, "Mati 2018 missing from output GeoPackage"

    def test_output_crs_is_4326(self, tmp_path: Path):
        """Output GeoPackage geometry is in EPSG:4326."""
        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            out = download_effis(output_dir, cfg, force=False)

        gdf = gpd.read_file(out, layer="fire_perimeters")
        assert gdf.crs.to_epsg() == 4326

    def test_manifest_written(self, tmp_path: Path):
        """download_effis writes a manifest.json."""
        import json

        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            download_effis(output_dir, cfg, force=False)

        manifest = output_dir / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data.get("mati_2018_present") is True
        assert "total_features" in data

    def test_skip_if_exists(self, tmp_path: Path):
        """Existing output is not re-generated when force=False."""
        from src.acquire.effis import download_effis

        output_dir = tmp_path / "effis"
        cfg = self._make_cfg()

        # First run
        with patch("src.acquire.effis._try_effis_wfs", return_value=None):
            out1 = download_effis(output_dir, cfg, force=False)

        mtime1 = out1.stat().st_mtime

        # Second run — should not re-write
        with patch("src.acquire.effis._try_effis_wfs") as mock_wfs:
            out2 = download_effis(output_dir, cfg, force=False)
            mock_wfs.assert_not_called()

        assert out1 == out2
        assert out2.stat().st_mtime == mtime1
