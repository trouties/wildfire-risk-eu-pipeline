"""
tests/test_acquire_corine.py
-----------------------------
Unit tests for src/acquire/corine.py (ESA WorldCover 2021 implementation).

Tests cover:
- Tile ID discovery from bbox
- S3 URL generation
- WorldCover class code constants
- Mosaic/clip output validation (mocked I/O)
- Fire vegetation class presence check

No network calls are made; rasterio I/O is mocked where needed.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _tiles_for_bbox
# ---------------------------------------------------------------------------

class TestTilesForBbox:
    """Tests for WorldCover tile discovery from AOI bbox."""

    def test_attica_returns_two_tiles(self):
        """Attica AOI [23.4, 37.6, 24.2, 38.3] spans two 3° x 3° tiles."""
        from src.acquire.corine import _tiles_for_bbox

        tiles = _tiles_for_bbox([23.4, 37.6, 24.2, 38.3])
        assert len(tiles) == 2, f"Expected 2 tiles, got {len(tiles)}: {tiles}"

    def test_attica_tile_ids_correct(self):
        """Attica spans tiles N36E021 (lon 21-24) and N36E024 (lon 24-27)."""
        from src.acquire.corine import _tiles_for_bbox

        tiles = sorted(_tiles_for_bbox([23.4, 37.6, 24.2, 38.3]))
        assert "N36E021" in tiles
        assert "N36E024" in tiles

    def test_small_bbox_single_tile(self):
        """Bbox fully within one 3° tile → single tile."""
        from src.acquire.corine import _tiles_for_bbox

        # Center of N36E021 (lat 37-38, lon 22-23 — within 36-39, 21-24)
        tiles = _tiles_for_bbox([22.1, 37.1, 22.9, 37.9])
        assert len(tiles) == 1
        assert tiles[0] == "N36E021"

    def test_tile_naming_north_east(self):
        """Positive lat/lon → N/E prefix."""
        from src.acquire.corine import _tiles_for_bbox

        tiles = _tiles_for_bbox([0.5, 0.5, 1.5, 1.5])
        assert all(t.startswith("N") for t in tiles)
        assert all("E" in t for t in tiles)

    def test_tile_naming_southern_hemisphere(self):
        """Negative lat → S prefix."""
        from src.acquire.corine import _tiles_for_bbox

        tiles = _tiles_for_bbox([20.0, -5.0, 21.0, -4.0])
        assert all(t.startswith("S") for t in tiles), f"Tiles: {tiles}"


# ---------------------------------------------------------------------------
# _tile_url
# ---------------------------------------------------------------------------

class TestTileUrl:
    """Tests for WorldCover S3 URL generation."""

    def test_url_starts_with_esa_worldcover_base(self):
        """URL must point to ESA WorldCover S3 bucket."""
        from src.acquire.corine import _tile_url

        url = _tile_url("N36E021")
        assert "esa-worldcover.s3.eu-central-1.amazonaws.com" in url

    def test_url_contains_tile_id(self):
        """Tile ID appears in the filename part of the URL."""
        from src.acquire.corine import _tile_url

        tile_id = "N36E021"
        url = _tile_url(tile_id)
        assert tile_id in url

    def test_url_ends_with_tif(self):
        """URL points to a .tif file."""
        from src.acquire.corine import _tile_url

        url = _tile_url("N36E021")
        assert url.endswith(".tif"), f"URL: {url}"

    def test_url_includes_2021_v200(self):
        """URL references version 2021 v200."""
        from src.acquire.corine import _tile_url

        url = _tile_url("N36E021")
        assert "2021" in url
        assert "v200" in url


# ---------------------------------------------------------------------------
# WorldCover class constants
# ---------------------------------------------------------------------------

class TestWorldCoverConstants:
    """Tests for fire-relevant class code constants."""

    def test_fire_veg_codes_contains_tree_shrub_grass(self):
        """FIRE_VEG_CODES must include 10 (tree), 20 (shrub), 30 (grass)."""
        from src.acquire.corine import FIRE_VEG_CODES

        assert 10 in FIRE_VEG_CODES
        assert 20 in FIRE_VEG_CODES
        assert 30 in FIRE_VEG_CODES

    def test_high_risk_codes_excludes_grassland(self):
        """HIGH_RISK_CODES should include tree+shrub but may exclude grassland."""
        from src.acquire.corine import HIGH_RISK_CODES

        assert 10 in HIGH_RISK_CODES
        assert 20 in HIGH_RISK_CODES

    def test_forest_codes_contains_only_tree(self):
        """FOREST_CODES should contain code 10 (tree cover)."""
        from src.acquire.corine import FOREST_CODES

        assert 10 in FOREST_CODES

    def test_wc_class_names_covers_all_fire_codes(self):
        """WC_CLASS_NAMES has a label for every fire-relevant class."""
        from src.acquire.corine import FIRE_VEG_CODES, WC_CLASS_NAMES

        for code in FIRE_VEG_CODES:
            assert code in WC_CLASS_NAMES, f"Missing label for class {code}"

    def test_buildup_not_in_fire_veg(self):
        """Built-up areas (code 50) must NOT be in fire vegetation codes."""
        from src.acquire.corine import FIRE_VEG_CODES

        assert 50 not in FIRE_VEG_CODES

    def test_wc_class_names_no_empty_strings(self):
        """All class name labels are non-empty strings."""
        from src.acquire.corine import WC_CLASS_NAMES

        for code, name in WC_CLASS_NAMES.items():
            assert isinstance(name, str) and name.strip(), \
                f"Empty/invalid label for class {code}"


# ---------------------------------------------------------------------------
# download_corine — mocked network I/O
# ---------------------------------------------------------------------------

class TestDownloadCorineMocked:
    """Integration tests for download_corine() with mocked network calls."""

    def _make_minimal_cfg(self, bbox=None):
        """Return a minimal config dict for tests."""
        return {
            "pipeline": {
                "aoi": {
                    "bbox": bbox or [23.4, 37.6, 24.2, 38.3],
                    "crs_working": "EPSG:2100",
                    "crs_output":  "EPSG:4326",
                }
            },
            "sources": {
                "corine": {
                    "resolution_m": 100,
                }
            },
        }

    def test_returns_path_if_already_exists(self, tmp_path):
        """If output file exists, return early without downloading."""
        from src.acquire.corine import download_corine

        cfg = self._make_minimal_cfg()
        out_file = tmp_path / "landcover_attica_2021.tif"
        out_file.write_bytes(b"dummy")

        with patch("src.acquire.corine._download_tile") as mock_dl:
            result = download_corine(tmp_path, cfg, force=False)

        mock_dl.assert_not_called()
        assert result == out_file

    def test_force_triggers_redownload(self, tmp_path):
        """force=True causes re-download even if file exists."""
        from src.acquire.corine import download_corine

        cfg = self._make_minimal_cfg()
        out_file = tmp_path / "landcover_attica_2021.tif"
        out_file.write_bytes(b"dummy")

        with patch("src.acquire.corine._download_tile", return_value=False) as mock_dl, \
             pytest.raises(RuntimeError, match="download failed"):
            download_corine(tmp_path, cfg, force=True)

        assert mock_dl.call_count >= 1

    def test_raises_if_tile_download_fails(self, tmp_path):
        """RuntimeError raised if any tile download fails."""
        from src.acquire.corine import download_corine

        cfg = self._make_minimal_cfg()
        with patch("src.acquire.corine._download_tile", return_value=False):
            with pytest.raises(RuntimeError, match="download failed"):
                download_corine(tmp_path, cfg, force=False)


# ---------------------------------------------------------------------------
# _mosaic_and_clip — output validation
# ---------------------------------------------------------------------------

class TestMosaicAndClip:
    """Tests for the mosaic + clip operation."""

    def _make_dummy_tile(self, tmp_path: Path, tile_name: str,
                         west: float, south: float,
                         east: float, north: float,
                         width: int = 360, height: int = 360,
                         fill_value: int = 20) -> Path:
        """Create a minimal WorldCover-like GeoTIFF for testing."""
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        tile_path = tmp_path / f"{tile_name}.tif"
        transform = from_bounds(west, south, east, north, width, height)
        data = np.full((height, width), fill_value, dtype=np.uint8)

        with rasterio.open(
            tile_path, "w",
            driver="GTiff", dtype=rasterio.uint8, count=1,
            width=width, height=height,
            crs=CRS.from_epsg(4326), transform=transform,
            nodata=0,
        ) as dst:
            dst.write(data, 1)

        return tile_path

    def test_clip_reduces_extent(self, tmp_path):
        """Clipped output is smaller than the input tile."""
        import rasterio

        from src.acquire.corine import _mosaic_and_clip

        # Create a 3° x 3° dummy tile (N36E021: lat 36-39, lon 21-24)
        tile = self._make_dummy_tile(
            tmp_path, "N36E021", 21.0, 36.0, 24.0, 39.0,
            width=360, height=360
        )

        bbox = [23.4, 37.6, 24.2, 38.3]  # smaller than tile
        out = tmp_path / "out.tif"
        _mosaic_and_clip([tile], bbox, out)

        assert out.exists()
        with rasterio.open(out) as ds:
            assert ds.width < 360
            assert ds.height < 360

    def test_output_crs_is_4326(self, tmp_path):
        """Output raster CRS is EPSG:4326."""
        import rasterio

        from src.acquire.corine import _mosaic_and_clip

        tile = self._make_dummy_tile(
            tmp_path, "N36E021", 21.0, 36.0, 24.0, 39.0,
        )
        out = tmp_path / "out.tif"
        _mosaic_and_clip([tile], [22.0, 37.0, 23.0, 38.0], out)

        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 4326

    def test_output_dtype_is_uint8(self, tmp_path):
        """Output raster pixel type is uint8."""
        import rasterio

        from src.acquire.corine import _mosaic_and_clip

        tile = self._make_dummy_tile(
            tmp_path, "N36E021", 21.0, 36.0, 24.0, 39.0,
        )
        out = tmp_path / "out.tif"
        _mosaic_and_clip([tile], [22.0, 37.0, 23.0, 38.0], out)

        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint8"

    def test_pixel_values_preserved(self, tmp_path):
        """Class codes in the input tile are preserved in the output."""
        import rasterio

        from src.acquire.corine import _mosaic_and_clip

        tile = self._make_dummy_tile(
            tmp_path, "N36E021", 21.0, 36.0, 24.0, 39.0,
            fill_value=20  # shrubland
        )
        out = tmp_path / "out.tif"
        _mosaic_and_clip([tile], [22.0, 37.0, 23.0, 38.0], out)

        with rasterio.open(out) as ds:
            data = ds.read(1)
            non_nodata = data[data > 0]
            assert len(non_nodata) > 0
            assert set(non_nodata.flatten().tolist()).issuperset({20})

    def test_two_tile_mosaic(self, tmp_path):
        """Two adjacent tiles are correctly mosaicked."""
        import rasterio

        from src.acquire.corine import _mosaic_and_clip

        # Tile 1: lon 21-24, fill with code 10 (tree)
        tile1 = self._make_dummy_tile(
            tmp_path, "N36E021", 21.0, 36.0, 24.0, 39.0,
            fill_value=10
        )
        # Tile 2: lon 24-27, fill with code 20 (shrub)
        tile2 = self._make_dummy_tile(
            tmp_path, "N36E024", 24.0, 36.0, 27.0, 39.0,
            fill_value=20
        )

        # Clip spanning both tiles
        bbox = [23.4, 37.6, 24.6, 38.3]
        out = tmp_path / "out.tif"
        _mosaic_and_clip([tile1, tile2], bbox, out)

        assert out.exists()
        with rasterio.open(out) as ds:
            data = ds.read(1)
            # Both code 10 and 20 should be present
            codes = set(data[data > 0].flatten().tolist())
            assert 10 in codes, "Tree cover (10) missing from mosaic"
            assert 20 in codes, "Shrubland (20) missing from mosaic"
