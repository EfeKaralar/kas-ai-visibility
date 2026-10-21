"""Collect the destination's accommodation from Google Maps via Serper.dev.

Grid of search points over the polygon x hotel keywords, paging each search until a
page adds no new places. Raw responses are cached per request; the deduplicated list
goes to data/processed/places.parquet.

    uv run python -m src.collect_places
"""

import math
import re

import pandas as pd
from shapely.geometry import Point
from unidecode import unidecode

from src.common import PROCESSED, SETTINGS, Budget, destination_polygon, haversine_km, km_from_center
from src.serper import Serper

STAGE = "place_collection"
# Booking-channel listings for single units, e.g. "Kale Apart - Studio with Sea View"
UNIT_LISTING = re.compile(r"\s-\s.*\b(apartment|studio|room|house|suite|villa|bedroom|daire|oda|view|balcony|"
                          r"loft|penthouse|bungalow)\b", re.IGNORECASE)
SAME_PLACE_KM = 0.3


def grid_points(polygon, spacing_km: float) -> list[tuple[float, float]]:
    """Grid over the polygon's bounding box, kept if within ~300 m of the polygon."""
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    lat_step = spacing_km / 111.0
    lon_step = spacing_km / (111.0 * math.cos(math.radians((min_lat + max_lat) / 2)))
    buffered = polygon.buffer(0.003)
    points, lat = [], min_lat
    while lat <= max_lat + 1e-9:
        lon = min_lon
        while lon <= max_lon + 1e-9:
            if buffered.contains(Point(lon, lat)):
                points.append((round(lat, 5), round(lon, 5)))
            lon += lon_step
        lat += lat_step
    return points


def normalize_name(name: str) -> str:
    return " ".join(unidecode(name or "").lower().replace("&", " ").split())


def parse_places(record: dict) -> list[dict]:
    rows = []
    for item in record["response"].get("places") or []:
        if item.get("latitude") is None:
            continue
        rows.append({
            "cid": item.get("cid"), "place_id": item.get("placeId"), "name": item.get("title"),
            "category": item.get("type"), "categories": item.get("types") or [],
            "address": item.get("address"), "lat": item["latitude"], "lon": item["longitude"],
            "rating": item.get("rating"), "review_count": item.get("ratingCount"),
            "website": item.get("website"), "keyword": record["request"]["q"],
        })
    return rows


def property_name(name: str) -> str:
    """Strip a unit description so unit listings share their property's name."""
    return name.split(" - ")[0].strip() if UNIT_LISTING.search(name or "") else name


def deduplicate(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge by Google cid, then merge entries with the same property name within 300 m.

    The second pass also folds single-unit listings into their property; the entry with
    the most reviews represents the property and the others are counted in unit_listings.
    """
    raw = raw.assign(key=raw["cid"].fillna(raw["place_id"]))
    groups = raw.groupby("key", sort=False)
    places = groups.first().reset_index()
    places["keywords"] = groups["keyword"].agg(lambda keywords: sorted(set(keywords))).values
    places["search_hits"] = groups.size().values

    places["listing_name"] = places["name"]
    places["name"] = places["name"].map(property_name)
    places["normalized_name"] = places["name"].map(normalize_name)
    places = places.sort_values("review_count", ascending=False, na_position="last")
    representative = {}
    kept = []
    for place in places.itertuples():
        twin = next((other for other in kept if other.normalized_name == place.normalized_name
                     and haversine_km(other.lat, other.lon, place.lat, place.lon) < SAME_PLACE_KM), None)
        if twin is None:
            kept.append(place)
        representative[place.key] = (twin or place).key
    places["representative"] = places["key"].map(representative)
    merged = places.groupby("representative")
    places = places[places["key"] == places["representative"]].set_index("key")
    places["unit_listings"] = merged.size() - 1
    places["keywords"] = merged["keywords"].agg(lambda lists: sorted({k for keywords in lists for k in keywords}))
    places["search_hits"] = merged["search_hits"].sum()
    print(f"Merged {int(places['unit_listings'].sum())} duplicate or single-unit listings into their property")
    return places.reset_index().drop(columns=["keyword", "representative"])


def main():
    config, budget = SETTINGS["place_collection"], Budget()
    client = Serper(STAGE, budget)
    polygon = destination_polygon()
    points = grid_points(polygon, config["grid_spacing_km"])
    print(f"{len(points)} grid points x {len(config['keywords'])} keywords, "
          f"up to {config['max_pages']} pages each")

    rows, saturation, seen = [], [], set()
    for index, (lat, lon) in enumerate(points, start=1):
        for keyword in config["keywords"]:
            for page in range(1, config["max_pages"] + 1):
                items = parse_places(client.maps_search(keyword, lat, lon, page))
                new = {item["cid"] or item["place_id"] for item in items} - seen
                seen |= new
                rows += items
                saturation.append({"search": len(saturation) + 1, "keyword": keyword, "lat": lat, "lon": lon,
                                   "page": page, "returned": len(items), "new": len(new),
                                   "cumulative_unique": len(seen)})
                if not items or not new:
                    break
        print(f"  point {index}/{len(points)}: {len(seen)} unique places, "
              f"{client.credits_used()} credits used")

    places = deduplicate(pd.DataFrame(rows))
    places["in_destination"] = [polygon.contains(Point(lon, lat)) for lat, lon in zip(places.lat, places.lon)]
    places["km_from_center"] = [round(km_from_center(lat, lon), 2) for lat, lon in zip(places.lat, places.lon)]
    places["place_uid"] = ["P" + str(number).zfill(4) for number in range(len(places))]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    places.to_parquet(PROCESSED / "places.parquet", index=False)
    pd.DataFrame(saturation).to_csv(PROCESSED / "place_search_saturation.csv", index=False)

    inside = places[places.in_destination]
    print(f"\nUnique places: {len(places)} (inside destination: {len(inside)})")
    print("Categories inside destination:")
    print(inside["category"].value_counts(dropna=False).to_string())
    print(f"With website: {inside.website.notna().sum()}, with rating: {inside.rating.notna().sum()}")
    print(f"Serper credits used: {client.credits_used()}")


if __name__ == "__main__":
    main()
