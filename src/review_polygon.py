"""Render the destination polygon over a map for approval.

Overlays OpenStreetMap accommodation (free, ODbL) and, once places have been
collected, the Google places. Output: data/review/polygon_map.html

    uv run python -m src.review_polygon
"""

import json

import folium
import pandas as pd
import requests

from src.common import PROCESSED, RAW, ROOT, SETTINGS, destination_polygon, in_destination

OSM_FILE = RAW / "osm" / "accommodation.json"
OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_osm() -> list[dict]:
    if not OSM_FILE.exists():
        query = (
            '[out:json][timeout:60];'
            'nwr["tourism"~"hotel|guest_house|apartment|hostel|motel"](36.15,29.52,36.25,29.72);'
            'out center tags;'
        )
        response = requests.post(OVERPASS, data={"data": query}, timeout=90,
                                 headers={"User-Agent": "kas-ai-visibility-poc/0.1 (research)"})
        response.raise_for_status()
        OSM_FILE.parent.mkdir(parents=True, exist_ok=True)
        OSM_FILE.write_text(response.text)
    rows = []
    for element in json.loads(OSM_FILE.read_text())["elements"]:
        position = element.get("center", element)
        rows.append({"name": element["tags"].get("name", "(unnamed)"), "type": element["tags"]["tourism"],
                     "lat": position["lat"], "lon": position["lon"]})
    return rows


def main():
    polygon = destination_polygon()
    center_lat, center_lon = SETTINGS["destination"]["center"]
    world_map = folium.Map(location=[center_lat, center_lon - 0.02], zoom_start=14)
    folium.GeoJson(json.loads((ROOT / SETTINGS["destination"]["polygon_file"]).read_text()),
                   style_function=lambda _: {"color": "#2563eb", "weight": 2, "fillOpacity": 0.08}
                   ).add_to(world_map)

    osm_places = fetch_osm()
    inside = 0
    for place in osm_places:
        is_inside = in_destination(place["lat"], place["lon"], polygon)
        inside += is_inside
        folium.CircleMarker([place["lat"], place["lon"]], radius=4, weight=1,
                            color="#16a34a" if is_inside else "#dc2626", fill=True,
                            tooltip=f"OSM {place['type']}: {place['name']}").add_to(world_map)

    places_file = PROCESSED / "places.parquet"
    if places_file.exists():
        for place in pd.read_parquet(places_file).itertuples():
            folium.CircleMarker([place.lat, place.lon], radius=3, weight=1, color="#7c3aed", fill=True,
                                tooltip=f"Google {place.category}: {place.name}").add_to(world_map)

    output = ROOT / "data" / "review" / "polygon_map.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    world_map.save(output)
    print(f"OSM accommodation: {len(osm_places)} total, {inside} inside polygon")
    print(f"Map: {output}")


if __name__ == "__main__":
    main()
