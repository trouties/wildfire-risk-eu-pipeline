"""
tests/test_acquire_dem.py
--------------------------
Unit tests for src/acquire/dem.py.

All tests operate on in-memory / mocked data — no network calls, no disk I/O
beyond tmp_path fixtures. Download functions are tested by mocking requests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _stac_discover_tiles
# ---------------------------------------------------------------------------

class TestStacDiscoverTiles:
    """Tests for STAC tile URL discovery."""

    def _make_stac_response(self, tile_ids: list[str]) -> dict:
        """Build a minimal Earth Search STAC response."""
        features = []
        for tid in tile_ids:
            features.append({
                "id": tid,
                "assets": {
                    "data": {
                        "href": f"s3://copernicus-dem-30m/{tid}/{tid}.tif",
                    }
                },
            })
        return {"features": features}

    def test_discovers_all_four_tiles(self):
        """STAC response with 4 tiles → dict with 4 entries."""
        from src.acquire.dem import _TILE_IDS, _stac_discover_tiles

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self._make_stac_response(_TILE_IDS)

        with patch("requests.post", return_value=mock_resp):
            result = _stac_discover_tiles([23.4, 37.6, 24.2, 38.3])

        assert len(result) == 4
        for tid in _TILE_IDS:
            assert tid in result

    def test_s3_url_converted_to_https(self):
        """s3://bucket/key hrefs are converted to eu-central-1 HTTPS URLs."""
        from src.acquire.dem import _TILE_IDS, _stac_discover_tiles

        tile_id = _TILE_IDS[0]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = self._make_stac_response([tile_id])

        with patch("requests.post", return_value=mock_resp):
            result = _stac_discover_tiles([23.4, 37.6, 24.2, 38.3])

        url = result[tile_id]
        assert url.startswith("https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/")
        assert tile_id in url
        assert url.endswith(".tif")

    def test_http_href_passed_through_unchanged(self):
        """HTTPS hrefs in STAC response are used as-is (no conversion)."""
        from src.acquire.dem import _stac_discover_tiles

        tile_id = "Copernicus_DSM_COG_10_N37_00_E023_00_DEM"
        https_href = f"https://cdn.example.com/{tile_id}.tif"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "features": [
                {"id": tile_id, "assets": {"data": {"href": https_href}}}
            ]
        }

        with patch("requests.post", return_value=mock_resp):
            result = _stac_discover_tiles([23.4, 37.6, 24.2, 38.3])

        assert result[tile_id] == https_href

    def test_empty_stac_response_returns_empty_dict(self):
        """STAC response with no features → empty dict (no crash)."""
        from src.acquire.dem import _stac_discover_tiles

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"features": []}

        with patch("requests.post", return_value=mock_resp):
            result = _stac_discover_tiles([23.4, 37.6, 24.2, 38.3])

        assert result == {}


# ---------------------------------------------------------------------------
# _download_tile_s3
# ---------------------------------------------------------------------------

class TestDownloadTileS3:
    """Tests for single-tile S3 download."""

    def test_existing_file_skipped_when_not_forced(self, tmp_path: Path):
        """If tile already exists and force=False, no HTTP request is made."""
        from src.acquire.dem import _download_tile_s3

        tile_id = "Copernicus_DSM_COG_10_N37_00_E023_00_DEM"
        out = tmp_path / f"{tile_id}.tif"
        out.write_bytes(b"fake tiff content")

        with patch("requests.get") as mock_get:
            result = _download_tile_s3(tile_id, "https://example.com/tile.tif", tmp_path, force=False)
            mock_get.assert_not_called()

        assert result == out

    def test_force_redownloads_existing_file(self, tmp_path: Path):
        """force=True triggers re-download even if file exists."""
        from src.acquire.dem import _download_tile_s3

        tile_id = "Copernicus_DSM_COG_10_N37_00_E023_00_DEM"
        out = tmp_path / f"{tile_id}.tif"
        out.write_bytes(b"old content")

        new_content = b"new tiff content"
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(new_content))}
        mock_resp.iter_content = MagicMock(return_value=[new_content])

        with patch("requests.get", return_value=mock_resp):
            result = _download_tile_s3(
                tile_id, "https://example.com/tile.tif", tmp_path, force=True
            )

        assert result == out
        assert out.read_bytes() == new_content

    def test_download_saves_file(self, tmp_path: Path):
        """Successful download creates a file with the expected content."""
        from src.acquire.dem import _download_tile_s3

        tile_id = "Copernicus_DSM_COG_10_N38_00_E023_00_DEM"
        content = b"\x49\x49\x2a\x00" + b"\x00" * 100  # minimal TIFF header

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(content))}
        mock_resp.iter_content = MagicMock(return_value=[content])

        with patch("requests.get", return_value=mock_resp):
            result = _download_tile_s3(
                tile_id, "https://example.com/tile.tif", tmp_path, force=False
            )

        assert result.exists()
        assert result.name == f"{tile_id}.tif"
        assert result.read_bytes() == content


# ---------------------------------------------------------------------------
# download_dem (integration-level, fully mocked)
# ---------------------------------------------------------------------------

class TestDownloadDem:
    """Tests for the main download_dem() entry point."""

    def _make_cfg(self) -> dict:
        return {
            "pipeline": {"aoi": {"bbox": [23.4, 37.6, 24.2, 38.3]}},
            "sources": {
                "dem": {
                    "output_dir": "data/raw/dem",
                    "tiles": ["N37_E023", "N37_E024", "N38_E023", "N38_E024"],
                }
            },
        }

    def test_skips_existing_tiles(self, tmp_path: Path):
        """If all 4 tiles exist and force=False, no HTTP calls are made."""
        from src.acquire.dem import _TILE_IDS, download_dem

        output_dir = tmp_path / "dem"
        output_dir.mkdir()

        # Create fake tile files
        for tid in _TILE_IDS:
            (output_dir / f"{tid}.tif").write_bytes(b"fake")

        cfg = self._make_cfg()

        with patch("requests.post") as mock_post, \
             patch("requests.get") as mock_get, \
             patch("rasterio.open") as mock_open:
            # Mock rasterio for verification
            mock_ds = MagicMock()
            mock_ds.__enter__ = lambda s: s
            mock_ds.__exit__ = MagicMock(return_value=False)
            mock_ds.crs = MagicMock(return_value="EPSG:4326")
            mock_ds.width = 3600
            mock_ds.height = 3600
            mock_ds.dtypes = ("float32",)
            mock_ds.nodata = -9999
            mock_ds.bounds = MagicMock(left=23.0, bottom=37.0, right=24.0, top=38.0)
            mock_open.return_value = mock_ds

            # STAC returns 4 tiles
            mock_post_resp = MagicMock()
            mock_post_resp.raise_for_status = MagicMock()
            mock_post_resp.json.return_value = {
                "features": [
                    {
                        "id": tid,
                        "assets": {"data": {"href": f"s3://copernicus-dem-30m/{tid}/{tid}.tif"}}
                    }
                    for tid in _TILE_IDS
                ]
            }
            mock_post.return_value = mock_post_resp

            paths = download_dem(output_dir, cfg, force=False)

        # GET (S3 download) should NOT have been called since files exist
        mock_get.assert_not_called()
        assert len(paths) == 4

    def test_manifest_written(self, tmp_path: Path):
        """download_dem writes a manifest.json in the output directory."""
        from src.acquire.dem import _TILE_IDS, download_dem

        output_dir = tmp_path / "dem"
        output_dir.mkdir()

        for tid in _TILE_IDS:
            (output_dir / f"{tid}.tif").write_bytes(b"x")

        cfg = self._make_cfg()

        # Minimal mocks
        with patch("requests.post") as mock_post, \
             patch("rasterio.open") as mock_open:
            mock_post_resp = MagicMock()
            mock_post_resp.raise_for_status = MagicMock()
            mock_post_resp.json.return_value = {"features": [
                {"id": tid, "assets": {"data": {"href": f"s3://bucket/{tid}.tif"}}}
                for tid in _TILE_IDS
            ]}
            mock_post.return_value = mock_post_resp

            mock_ds = MagicMock()
            mock_ds.__enter__ = lambda s: s
            mock_ds.__exit__ = MagicMock(return_value=False)
            mock_ds.crs = "EPSG:4326"
            mock_ds.width = 3600
            mock_ds.height = 3600
            mock_ds.dtypes = ("float32",)
            mock_ds.nodata = -9999
            mock_ds.bounds = MagicMock(left=23.0, bottom=37.0, right=24.0, top=38.0)
            mock_open.return_value = mock_ds

            download_dem(output_dir, cfg, force=False)

        manifest = output_dir / "manifest.json"
        assert manifest.exists()
        import json
        data = json.loads(manifest.read_text())
        assert "source" in data
        assert "tiles" in data
        assert "aoi_bbox_wsen" in data

    def test_raises_when_all_downloads_fail(self, tmp_path: Path):
        """download_dem raises RuntimeError if no tiles are retrieved."""
        from src.acquire.dem import download_dem

        output_dir = tmp_path / "dem"
        output_dir.mkdir()
        cfg = self._make_cfg()

        import requests as req

        with patch("requests.post", side_effect=req.exceptions.ConnectionError("no network")), \
             patch("requests.get", side_effect=req.exceptions.ConnectionError("no network")):

            # CDSE fallback also needs to fail
            with patch("src.acquire.dem._get_cdse_token",
                       side_effect=RuntimeError("no cred")):
                with pytest.raises(RuntimeError, match="DEM download failed"):
                    download_dem(output_dir, cfg, force=False)


# ---------------------------------------------------------------------------
# _TILE_IDS constant sanity check
# ---------------------------------------------------------------------------

class TestTileIds:
    """Sanity checks for the _TILE_IDS constant."""

    def test_four_tiles(self):
        from src.acquire.dem import _TILE_IDS
        assert len(_TILE_IDS) == 4

    def test_tile_naming_convention(self):
        """All tile IDs follow the Copernicus DSM COG naming pattern."""
        import re

        from src.acquire.dem import _TILE_IDS
        pattern = re.compile(r"^Copernicus_DSM_COG_10_N\d{2}_00_E\d{3}_00_DEM$")
        for tid in _TILE_IDS:
            assert pattern.match(tid), f"Unexpected tile ID: {tid}"

    def test_covers_attica_bbox(self):
        """Tile IDs span the correct latitude/longitude cells for Attica."""
        # Parse lat/lon from tile name: N37_00_E023_00 → lat=37, lon=23
        import re

        from src.acquire.dem import _TILE_IDS
        lats, lons = set(), set()
        for tid in _TILE_IDS:
            m = re.search(r"N(\d{2})_00_E(\d{3})_00", tid)
            assert m, f"Cannot parse lat/lon from {tid}"
            lats.add(int(m.group(1)))
            lons.add(int(m.group(2)))

        assert 37 in lats and 38 in lats, "Missing N37 or N38 tile"
        assert 23 in lons and 24 in lons, "Missing E023 or E024 tile"
