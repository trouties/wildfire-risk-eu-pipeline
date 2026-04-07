"""
risk_map.py — WildfireRisk-EU Output: Interactive Folium Risk Map

Produces an HTML map with:
  - All buildings colored by risk_class (1=green → 5=red)
  - Per-event EFFIS fire perimeter overlays for all LOEO validation events
  - Burned buildings per event highlighted with distinct colors
  - Validation bounding boxes per event

Size optimization:
  Buildings are rendered via injected JavaScript from a compact JSON data
  array (~25 bytes/building) rather than individual Folium CircleMarker
  objects (~400 bytes/building).  This reduces a 226K-building map from
  ~107 MB to ~10-15 MB.

Geometry notes:
  - Buildings: centroid_lat/lon already in WGS84 from DuckDB
  - EFFIS perimeters: WKT stored in EPSG:2100 (Greek Grid); converted to
    WGS84 via pyproj before passing to Folium

Output: outputs/maps/wildfire_risk_map.html
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import folium
import yaml
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "wildfire_risk.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs" / "maps"
OUT_FILE = OUT_DIR / "wildfire_risk_map.html"
CONFIG_PATH = PROJECT_ROOT / "config" / "validation.yaml"

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

# Per-event display config
EVENT_STYLES = {
    "kalamos_2015":  {"dot": "#1a237e", "fill": "#3949ab", "stroke": "#1a237e", "label": "Kalamos 2015"},
    "mati_2018":     {"dot": "#000000", "fill": "#ff8c00", "stroke": "#cc4400", "label": "Mati 2018"},
    "varybobi_2021": {"dot": "#6a1b9a", "fill": "#ab47bc", "stroke": "#6a1b9a", "label": "Varybobi 2021"},
    "acharnes_2021": {"dot": "#bf360c", "fill": "#ff7043", "stroke": "#bf360c", "label": "Acharnes 2021"},
}


# ---------- helpers ----------------------------------------------------------

def _epsg2100_to_wgs84_geojson(wkt_str: str) -> dict:
    """Convert a WKT geometry in EPSG:2100 to a GeoJSON dict in WGS84."""
    transformer = Transformer.from_crs("EPSG:2100", "EPSG:4326", always_xy=True)
    geom = shapely_wkt.loads(wkt_str)
    geom_wgs84 = shapely_transform(transformer.transform, geom)
    return json.loads(json.dumps(mapping(geom_wgs84)))


def _load_events() -> dict:
    """Load event definitions from validation config."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("holdout_events", {})


# ---------- main -------------------------------------------------------------

def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = _load_events()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # All buildings with scores (round coords to reduce data size)
    df_all = con.execute("""
        SELECT r.building_id,
               round(b.centroid_lat, 4) AS centroid_lat,
               round(b.centroid_lon, 4) AS centroid_lon,
               r.risk_class
        FROM risk_scores r
        JOIN buildings b USING (building_id)
    """).df()

    # Burned buildings per event
    df_burned = con.execute("""
        SELECT v.building_id, v.event_id,
               round(b.centroid_lat, 5) AS centroid_lat,
               round(b.centroid_lon, 5) AS centroid_lon,
               v.risk_class
        FROM validation_results v
        JOIN buildings b USING (building_id)
        WHERE v.burned = TRUE
    """).df()

    # EFFIS perimeters for validation events
    event_fire_ids = {k: v["fire_id"] for k, v in events.items()}
    fire_id_list = ", ".join(f"'{fid}'" for fid in event_fire_ids.values())
    perimeters = {}
    for row in con.execute(
        f"SELECT fire_id, geometry FROM effis_perimeters WHERE fire_id IN ({fire_id_list})"
    ).fetchall():
        perimeters[row[0]] = row[1]

    con.close()

    total_burned = len(df_burned)
    print(f"[risk_map] Loaded {len(df_all):,} buildings, {total_burned:,} burned across {len(events)} events")

    # --- Compute map center from all validation bboxes ---
    all_bboxes = [v["validation_bbox"] for v in events.values() if "validation_bbox" in v]
    if all_bboxes:
        center_lat = sum((b[1] + b[3]) / 2 for b in all_bboxes) / len(all_bboxes)
        center_lon = sum((b[0] + b[2]) / 2 for b in all_bboxes) / len(all_bboxes)
    else:
        center_lat, center_lon = 38.05, 23.85

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # --- Build compact data arrays for bulk JavaScript rendering ---
    # Instead of creating 226K+ individual Folium CircleMarker objects
    # (each ~400 bytes of HTML/JS), we inject a single compact JSON array
    # and a JavaScript loop that creates CircleMarkers at runtime.
    # This reduces the HTML from ~107 MB to ~10-15 MB.

    # Group buildings by risk class → list of [lat, lon]
    buildings_by_class: dict[int, list] = {cls: [] for cls in range(1, 6)}
    for row in df_all.itertuples(index=False):
        buildings_by_class[row.risk_class].append([row.centroid_lat, row.centroid_lon])

    # Color map as JSON for JavaScript
    colors_js = json.dumps({str(k): v for k, v in RISK_COLORS.items()})
    labels_js = json.dumps({str(k): v for k, v in RISK_LABELS.items()})

    # Build JS code to create all building markers
    js_parts = []
    js_parts.append(f"var riskColors = {colors_js};")
    js_parts.append(f"var riskLabels = {labels_js};")

    for cls in [5, 4, 3, 2, 1]:
        coords = buildings_by_class[cls]
        coords_json = json.dumps(coords, separators=(",", ":"))
        var_name = f"bldgData{cls}"
        layer_var = f"layer{cls}"
        js_parts.append(f"var {var_name} = {coords_json};")
        js_parts.append(f"""
var {layer_var} = L.featureGroup();
for (var i = 0; i < {var_name}.length; i++) {{
  L.circleMarker({var_name}[i], {{
    radius: 2, color: riskColors["{cls}"], fill: true,
    fillColor: riskColors["{cls}"], fillOpacity: 0.7, weight: 0
  }}).addTo({layer_var});
}}
{layer_var}.addTo(map);
layerControl.addOverlay({layer_var}, "Risk Class {cls} — " + riskLabels["{cls}"] + " (" + {var_name}.length.toLocaleString() + ")");
""")

    # --- Burned buildings per event (few markers, use Folium directly) ---
    legend_events = []
    burned_js_parts = []
    for event_key, event_cfg in events.items():
        style = EVENT_STYLES.get(event_key, {"dot": "#333", "fill": "#999", "stroke": "#666",
                                              "label": event_key})
        fire_id = event_cfg["fire_id"]
        event_label = style["label"]
        dot_color = style["dot"]

        df_evt = df_burned[df_burned["event_id"] == fire_id]
        n_burned = len(df_evt)

        # Burned buildings as compact JS array
        burned_coords = []
        for row in df_evt.itertuples(index=False):
            burned_coords.append([row.centroid_lat, row.centroid_lon, row.risk_class,
                                  row.building_id])

        burned_json = json.dumps(burned_coords, separators=(",", ":"))
        safe_key = event_key.replace("-", "_")
        burned_js_parts.append(f"""
var burnedData_{safe_key} = {burned_json};
var burnedLayer_{safe_key} = L.featureGroup();
for (var i = 0; i < burnedData_{safe_key}.length; i++) {{
  var d = burnedData_{safe_key}[i];
  L.circleMarker([d[0], d[1]], {{
    radius: 3, color: "{dot_color}", fill: true,
    fillColor: "{dot_color}", fillOpacity: 0.85, weight: 1
  }}).bindTooltip(d[3] + " | BURNED ({event_label}) | Class " + d[2]).addTo(burnedLayer_{safe_key});
}}
burnedLayer_{safe_key}.addTo(map);
layerControl.addOverlay(burnedLayer_{safe_key}, "Burned — {event_label} ({n_burned:,})");
""")

        # Fire perimeter (keep as Folium GeoJson — few objects)
        wkt = perimeters.get(fire_id)
        if wkt:
            try:
                geojson = _epsg2100_to_wgs84_geojson(wkt)
                fill_c = style["fill"]
                stroke_c = style["stroke"]
                folium.GeoJson(
                    {"type": "Feature", "geometry": geojson, "properties": {}},
                    name=f"{event_label} perimeter",
                    style_function=lambda _, fc=fill_c, sc=stroke_c: {
                        "fillColor": fc,
                        "color": sc,
                        "weight": 2.5,
                        "fillOpacity": 0.15,
                    },
                    tooltip=f"{event_label} fire perimeter ({fire_id})",
                ).add_to(m)
            except Exception as exc:
                print(f"[risk_map] WARNING: could not render {event_label} perimeter: {exc}")

        # Validation bbox
        bbox = event_cfg.get("validation_bbox")
        if bbox:
            west, south, east, north = bbox
            folium.Rectangle(
                bounds=[[south, west], [north, east]],
                color=style["stroke"],
                weight=2,
                fill=False,
                dash_array="8 4",
                tooltip=f"{event_label} validation bbox",
            ).add_to(m)

        legend_events.append((dot_color, f"Burned — {event_label}"))

    # --- Layer control (added early so JS can reference it) ---
    # We add a manual LayerControl via JS since we need to add layers dynamically
    # Don't use folium.LayerControl — we manage it in JS

    # --- Legend ---
    event_legend_items = "".join(
        f'<span style="color:{c}">&#9679;</span> {lbl}<br>' for c, lbl in legend_events
    )
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-size:12px; line-height:1.6;">
      <b>Wildfire Risk Class</b><br>
      <span style="color:#d73027">&#9679;</span> 5 — Very High<br>
      <span style="color:#fc8d59">&#9679;</span> 4 — High<br>
      <span style="color:#fee08b">&#9679;</span> 3 — Medium<br>
      <span style="color:#91cf60">&#9679;</span> 2 — Low<br>
      <span style="color:#1a9850">&#9679;</span> 1 — Very Low<br>
      <hr style="margin:4px 0">
      <b>Validation Events</b><br>
      {event_legend_items}
      <span style="color:#666">&#9633;</span> Validation bboxes
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Inject bulk marker JavaScript ---
    # Folium assigns a variable name to the map object. We need to find it.
    # The map variable name is accessible via m.get_name()
    map_var = m.get_name()

    all_js = "\n".join(js_parts + burned_js_parts)
    # Replace 'map' with the actual Folium map variable name
    bulk_js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        // Get reference to the Folium map object
        var map = {map_var};
        var layerControl = L.control.layers(null, {{}}, {{collapsed: false}}).addTo(map);

        // Add Folium-managed layers to the layer control
        map.eachLayer(function(layer) {{
            if (layer._name) {{
                layerControl.addOverlay(layer, layer._name);
            }}
        }});

        {all_js}
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(bulk_js))

    m.save(str(OUT_FILE))

    # Report file size
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"[risk_map] Saved map -> {OUT_FILE} ({size_mb:.1f} MB)")
    return OUT_FILE


if __name__ == "__main__":
    main()
