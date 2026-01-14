"""
tests/test_acquire_buildings.py
--------------------------------
Unit tests for src/acquire/buildings.py.

All tests operate on in-memory synthetic data — no network calls, no disk I/O,
no dependency on data/raw/. The acquisition logic that contacts Overpass API
is tested indirectly via the parser functions (_way_to_polygon, _parse_elements).
"""

from shapely.geometry import Polygon

from src.acquire.buildings import (
    _classify_use,
    _parse_elements,
    _way_to_polygon,
)

# ---------------------------------------------------------------------------
# _way_to_polygon
# ---------------------------------------------------------------------------

class TestWayToPolygon:
    """Tests for the raw geometry converter."""

    def _make_geom(self, lons, lats):
        return [{"lon": lo, "lat": la} for lo, la in zip(lons, lats)]

    def test_valid_closed_ring(self):
        """Standard closed 4-point ring → valid Polygon."""
        geom = self._make_geom(
            [23.7, 23.71, 23.71, 23.7,  23.7],
            [37.97, 37.97, 37.98, 37.98, 37.97],
        )
        poly = _way_to_polygon(geom)
        assert poly is not None
        assert isinstance(poly, Polygon)
        assert poly.is_valid
        assert not poly.is_empty

    def test_auto_closes_open_ring(self):
        """Open ring (first ≠ last) is force-closed and still returns a polygon."""
        geom = self._make_geom(
            [23.7, 23.71, 23.71, 23.7],
            [37.97, 37.97, 37.98, 37.98],
        )
        poly = _way_to_polygon(geom)
        assert poly is not None
        assert isinstance(poly, Polygon)

    def test_too_few_nodes_returns_none(self):
        """3 nodes (< 4) → None (degenerate)."""
        geom = self._make_geom([23.7, 23.71, 23.7], [37.97, 37.97, 37.97])
        assert _way_to_polygon(geom) is None

    def test_empty_list_returns_none(self):
        assert _way_to_polygon([]) is None

    def test_none_returns_none(self):
        assert _way_to_polygon(None) is None

    def test_degenerate_collinear_polygon_skipped(self):
        """All collinear nodes → zero-area polygon → None."""
        geom = self._make_geom(
            [23.7, 23.71, 23.72, 23.73, 23.7],
            [37.97, 37.97, 37.97, 37.97, 37.97],
        )
        # This produces a zero-area line-like geometry
        result = _way_to_polygon(geom)
        # Either None or empty — must not raise
        assert result is None or result.is_empty

    def test_large_polygon_area_positive(self):
        """Polygon area must be positive."""
        geom = self._make_geom(
            [23.7, 23.8, 23.8, 23.7, 23.7],
            [37.9, 37.9, 38.0, 38.0, 37.9],
        )
        poly = _way_to_polygon(geom)
        assert poly is not None
        assert poly.area > 0


# ---------------------------------------------------------------------------
# _classify_use
# ---------------------------------------------------------------------------

class TestClassifyUse:
    """Tests for the building use classification helper."""

    def test_known_residential_tags(self):
        for tag in ("house", "detached", "apartments", "residential", "villa"):
            assert _classify_use(tag, {}) == "residential", f"tag={tag}"

    def test_known_commercial_tags(self):
        for tag in ("commercial", "retail", "office", "hotel"):
            assert _classify_use(tag, {}) == "commercial", f"tag={tag}"

    def test_known_civic_tags(self):
        for tag in ("school", "hospital", "church", "fire_station"):
            assert _classify_use(tag, {}) == "civic", f"tag={tag}"

    def test_known_industrial_tags(self):
        for tag in ("industrial", "warehouse", "factory"):
            assert _classify_use(tag, {}) == "industrial", f"tag={tag}"

    def test_generic_yes_is_unknown(self):
        assert _classify_use("yes", {}) == "unknown"

    def test_unknown_tag_is_unknown(self):
        assert _classify_use("pavilion", {}) == "unknown"

    def test_amenity_fallback_for_yes_tag(self):
        """building=yes + amenity=hospital → civic via fallback."""
        assert _classify_use("yes", {"amenity": "hospital"}) == "civic"

    def test_amenity_fallback_fire_station(self):
        assert _classify_use("yes", {"amenity": "fire_station"}) == "civic"

    def test_amenity_ignored_when_building_tag_specific(self):
        """Explicit building tag takes priority over amenity."""
        assert _classify_use("commercial", {"amenity": "hospital"}) == "commercial"

    def test_case_insensitive(self):
        """Tags should be matched case-insensitively."""
        assert _classify_use("House", {}) == "residential"
        assert _classify_use("APARTMENTS", {}) == "residential"


# ---------------------------------------------------------------------------
# _parse_elements
# ---------------------------------------------------------------------------

class TestParseElements:
    """Tests for the Overpass element parser."""

    def _make_way(self, osm_id=12345, building="yes", name=None, extra_nodes=None):
        """Create a minimal valid Overpass way element."""
        base_coords = [
            {"lat": 37.97,  "lon": 23.70},
            {"lat": 37.97,  "lon": 23.71},
            {"lat": 37.971, "lon": 23.71},
            {"lat": 37.971, "lon": 23.70},
            {"lat": 37.97,  "lon": 23.70},   # closed
        ]
        if extra_nodes:
            base_coords = extra_nodes
        tags = {"building": building}
        if name:
            tags["name"] = name
        return {
            "type": "way",
            "id": osm_id,
            "geometry": base_coords,
            "tags": tags,
        }

    def test_single_valid_way(self):
        elements = [self._make_way(osm_id=1, building="house")]
        rows = _parse_elements(elements, "test_chunk")
        assert len(rows) == 1
        row = rows[0]
        assert row["building_id"] == "way/1"
        assert row["osm_id"] == 1
        assert row["osm_type"] == "way"
        assert row["source"] == "osm"
        assert row["building_tag"] == "house"
        assert row["building_use"] == "residential"
        assert row["chunk"] == "test_chunk"
        assert row["geometry"] is not None

    def test_name_preserved(self):
        elements = [self._make_way(osm_id=2, name="Test Building")]
        rows = _parse_elements(elements, "chunk_a")
        assert rows[0]["name"] == "Test Building"

    def test_missing_name_is_none(self):
        elements = [self._make_way(osm_id=3)]
        rows = _parse_elements(elements, "chunk_a")
        assert rows[0]["name"] is None

    def test_non_way_elements_skipped(self):
        """Node and relation elements should be skipped silently."""
        elements = [
            {"type": "node", "id": 999, "lat": 37.97, "lon": 23.70},
            {"type": "relation", "id": 888, "members": [], "tags": {"building": "yes"}},
            self._make_way(osm_id=5),
        ]
        rows = _parse_elements(elements, "chunk_a")
        assert len(rows) == 1
        assert rows[0]["building_id"] == "way/5"

    def test_degenerate_way_skipped(self):
        """Way with only 2 nodes → geometry is None → skipped."""
        bad_way = {
            "type": "way",
            "id": 77,
            "geometry": [{"lat": 37.97, "lon": 23.70}, {"lat": 37.971, "lon": 23.71}],
            "tags": {"building": "yes"},
        }
        rows = _parse_elements([bad_way], "chunk_a")
        assert len(rows) == 0

    def test_multiple_ways(self):
        elements = [self._make_way(i, "apartments") for i in range(10)]
        rows = _parse_elements(elements, "chunk_a")
        assert len(rows) == 10
        ids = {r["building_id"] for r in rows}
        assert len(ids) == 10   # all unique

    def test_empty_elements(self):
        rows = _parse_elements([], "chunk_a")
        assert rows == []

    def test_building_tag_passthrough(self):
        """building_tag should contain the raw OSM value."""
        elements = [self._make_way(10, building="apartments")]
        rows = _parse_elements(elements, "chunk_a")
        assert rows[0]["building_tag"] == "apartments"

    def test_no_building_tag_defaults_to_yes(self):
        """Ways without explicit building= tag get 'yes' as default."""
        el = {
            "type": "way",
            "id": 20,
            "geometry": [
                {"lat": 37.97, "lon": 23.70},
                {"lat": 37.97, "lon": 23.71},
                {"lat": 37.971, "lon": 23.71},
                {"lat": 37.971, "lon": 23.70},
                {"lat": 37.97, "lon": 23.70},
            ],
            "tags": {},   # no building tag
        }
        rows = _parse_elements([el], "chunk_a")
        assert len(rows) == 1
        assert rows[0]["building_tag"] == "yes"
        assert rows[0]["building_use"] == "unknown"


# ---------------------------------------------------------------------------
# Integration: config-driven chunk bbox access
# ---------------------------------------------------------------------------

class TestConfigChunks:
    """Verify that the config correctly exposes chunk bboxes."""

    def test_chunks_present(self, cfg):
        chunks = cfg["sources"]["buildings"]["primary"]["download_chunks"]
        assert "east_attica_ne" in chunks
        assert "east_attica_se" in chunks
        assert "north_attica_wui" in chunks

    def test_chunk_bbox_has_four_coords(self, cfg):
        chunks = cfg["sources"]["buildings"]["primary"]["download_chunks"]
        for name, vals in chunks.items():
            bbox = vals["bbox"]
            assert len(bbox) == 4, f"chunk '{name}' bbox must have [W,S,E,N]"

    def test_mati_inside_ne_chunk(self, cfg):
        """Mati (~24.0°E, 38.06°N) must be inside the east_attica_ne chunk."""
        bbox = cfg["sources"]["buildings"]["primary"]["download_chunks"][
            "east_attica_ne"
        ]["bbox"]
        w, s, e, n = bbox
        mati_lon, mati_lat = 23.978, 38.058
        assert w <= mati_lon <= e, "Mati longitude not in east_attica_ne chunk"
        assert s <= mati_lat <= n, "Mati latitude not in east_attica_ne chunk"

    def test_min_buildings_threshold(self, cfg):
        assert cfg["pipeline"]["run"]["min_buildings"] == 5000
