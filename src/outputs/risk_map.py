"""
risk_map.py — WildfireRisk-EU Output: Interactive Folium Risk Map (Task T12)

Produces an HTML map with:
  - All 84,767 buildings colored by risk_class (1=green → 5=red)
  - Mati 2018 EFFIS fire perimeter overlay (orange polygon)
  - Burned buildings highlighted in black
  - Validation bounding box outline

Geometry notes:
  - Buildings: centroid_lat/lon already in WGS84 from DuckDB
  - EFFIS perimeter: WKT stored in EPSG:2100 (Greek Grid); converted to
    WGS84 via pyproj before passing to Folium

Output: outputs/maps/wildfire_risk_map.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import folium
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "wildfire_risk.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs" / "maps"
OUT_FILE = OUT_DIR / "wildfire_risk_map.html"

# Validation bounding box [west, south, east, north] in WGS84
VALIDATION_BBOX = [23.85, 37.98, 24.10, 38.12]

# Risk class colours (green→red)
RISK_COLORS = {
    1: "#1a9850",  # Very Low  — green
    2: "#91cf60",  # Low       — light green
    3: "#fee08b",  # Medium    — yellow
    4: "#fc8d59",  # High      — orange
    5: "#d73027",  # Very High — red
}

RISK_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very High",
}


# ---------- helpers ----------------------------------------------------------

def _epsg2100_to_wgs84_geojson(wkt_str: str) -> dict:
    """Convert a WKT geometry in EPSG:2100 to a GeoJSON dict in WGS84."""
    transformer = Transformer.from_crs("EPSG:2100", "EPSG:4326", always_xy=True)
    geom = shapely_wkt.loads(wkt_str)
    geom_wgs84 = shapely_transform(transformer.transform, geom)
    return json.loads(json.dumps(mapping(geom_wgs84)))


def _buildings_geojson(df: pd.DataFrame) -> dict:
    """Convert a DataFrame with centroid_lat/lon to a GeoJSON FeatureCollection."""
    features = []
    for row in df.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.centroid_lon, row.centroid_lat],
            },
            "properties": {
                "building_id": row.building_id,
                "risk_class": int(row.risk_class),
                "composite_score": round(float(row.composite_score), 3),
                "burned": bool(getattr(row, "burned", False)),
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ---------- main -------------------------------------------------------------

def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # All buildings with scores
    df_all = con.execute("""
        SELECT r.building_id, b.centroid_lat, b.centroid_lon,
               r.composite_score, r.risk_class
        FROM risk_scores r
        JOIN buildings b USING (building_id)
    """).df()

    # Burned buildings (from validation results)
    df_burned = con.execute("""
        SELECT v.building_id, b.centroid_lat, b.centroid_lon,
               v.composite_score, v.risk_class
        FROM validation_results v
        JOIN buildings b USING (building_id)
        WHERE v.burned = TRUE
    """).df()

    # Mati 2018 perimeter WKT
    mati_row = con.execute(
        "SELECT geometry FROM effis_perimeters WHERE fire_id = 'EFFIS_20180723_009'"
    ).fetchone()
    con.close()

    if mati_row is None:
        sys.exit("ERROR risk_map: Mati perimeter not found in effis_perimeters")

    mati_wkt = mati_row[0]

    print(f"[risk_map] Loaded {len(df_all):,} buildings, {len(df_burned):,} burned")

    # --- Build map ---
    center_lat = (VALIDATION_BBOX[1] + VALIDATION_BBOX[3]) / 2
    center_lon = (VALIDATION_BBOX[0] + VALIDATION_BBOX[2]) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # --- Risk class layers (one per class for legend clarity) ---
    for cls in [5, 4, 3, 2, 1]:
        df_cls = df_all[df_all["risk_class"] == cls]
        color = RISK_COLORS[cls]
        label = RISK_LABELS[cls]
        layer = folium.FeatureGroup(name=f"Risk Class {cls} — {label}", show=True)
        for row in df_cls.itertuples(index=False):
            folium.CircleMarker(
                location=[row.centroid_lat, row.centroid_lon],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=0,
                tooltip=f"{row.building_id} | Class {row.risk_class} | Score {row.composite_score:.2f}",
            ).add_to(layer)
        layer.add_to(m)

    # --- Burned buildings overlay (black) ---
    burned_layer = folium.FeatureGroup(name="Burned buildings (Mati 2018)", show=True)
    for row in df_burned.itertuples(index=False):
        folium.CircleMarker(
            location=[row.centroid_lat, row.centroid_lon],
            radius=3,
            color="#000000",
            fill=True,
            fill_color="#000000",
            fill_opacity=0.85,
            weight=1,
            tooltip=f"{row.building_id} | BURNED | Class {row.risk_class}",
        ).add_to(burned_layer)
    burned_layer.add_to(m)

    # --- Mati 2018 perimeter (EPSG:2100 → WGS84) ---
    try:
        mati_geojson = _epsg2100_to_wgs84_geojson(mati_wkt)
        folium.GeoJson(
            {"type": "Feature", "geometry": mati_geojson, "properties": {}},
            name="Mati 2018 fire perimeter",
            style_function=lambda _: {
                "fillColor": "#ff8c00",
                "color": "#cc4400",
                "weight": 2.5,
                "fillOpacity": 0.15,
            },
            tooltip="Mati 2018 fire perimeter (EFFIS_20180723_009)",
        ).add_to(m)
    except Exception as exc:
        print(f"[risk_map] WARNING: could not render Mati perimeter: {exc}")

    # --- Validation bounding box ---
    west, south, east, north = VALIDATION_BBOX
    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#2255aa",
        weight=2,
        fill=False,
        dash_array="8 4",
        tooltip="Validation bounding box",
    ).add_to(m)

    # --- Layer control + legend ---
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-size:12px; line-height:1.6;">
      <b>Wildfire Risk Class</b><br>
      <span style="color:#d73027">&#9679;</span> 5 — Very High<br>
      <span style="color:#fc8d59">&#9679;</span> 4 — High<br>
      <span style="color:#fee08b">&#9679;</span> 3 — Medium<br>
      <span style="color:#91cf60">&#9679;</span> 2 — Low<br>
      <span style="color:#1a9850">&#9679;</span> 1 — Very Low<br>
      <span style="color:#000000">&#9679;</span> Burned (Mati 2018)<br>
      <span style="color:#2255aa">&#9633;</span> Validation bbox
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(OUT_FILE))
    print(f"[risk_map] Saved map -> {OUT_FILE}")
    return OUT_FILE


if __name__ == "__main__":
    main()
