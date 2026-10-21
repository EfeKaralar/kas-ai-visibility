"""Collect the destination's accommodation from Google Maps via DataForSEO.

Grid of search points over the polygon x hotel keywords. Raw responses are cached
per request; the deduplicated list goes to data/processed/places.parquet.

    uv run python -m src.collect_places
"""

import math

import pandas as pd
import requests
from shapely.geometry import Point
from unidecode import unidecode

from src.common import (PRICING, PROCESSED, RAW, SETTINGS, Budget, cache_get, cache_put,
                        destination_polygon, haversine_km, km_from_center, now_iso, request_hash,
                        require_env)

ENDPOINT = "https://api.dataforseo.com/v3/serp/google/maps/live/advanced"
CACHE = RAW / "dataforseo" / "maps"
STAGE = "place_collection"


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


def maps_search(keyword: str, lat: float, lon: float, auth: tuple, budget: Budget) -> dict:
    config = SETTINGS["place_collection"]
    task = {"keyword": keyword, "location_coordinate": f"{lat},{lon},{config['zoom']}",
            "language_code": config["language_code"], "depth": config["depth"], "device": "desktop"}
    key = request_hash(task)
    if cached := cache_get(CACHE, key):
        return cached
    budget.check(PRICING["dataforseo"]["maps_live_per_call"])
    response = requests.post(ENDPOINT, json=[task], auth=auth, timeout=120)
    response.raise_for_status()
    result = response.json()["tasks"][0]
    if result["status_code"] != 20000:
        raise RuntimeError(f"DataForSEO task failed: {result['status_code']} {result['status_message']}")
    budget.record(STAGE, "dataforseo", result.get("cost") or PRICING["dataforseo"]["maps_live_per_call"],
                  f"{keyword} @ {lat},{lon}")
    record = {"request": task, "fetched_at": now_iso(), "response": result}
    cache_put(CACHE, key, record)
    return record


def normalize_name(name: str) -> str:
    return " ".join(unidecode(name or "").lower().replace("&", " ").split())


def parse_items(record: dict) -> list[dict]:
    rows = []
    for result in record["response"].get("result") or []:
        for item in result.get("items") or []:
            if item.get("type") != "maps_search" or item.get("latitude") is None:
                continue
            rating = item.get("rating") or {}
            rows.append({
                "cid": item.get("cid"), "place_id": item.get("place_id"), "name": item.get("title"),
                "category": item.get("category"), "additional_categories": item.get("additional_categories") or [],
                "address": item.get("address"), "lat": item["latitude"], "lon": item["longitude"],
                "rating": rating.get("value"), "review_count": rating.get("votes_count"),
                "website": item.get("url"), "domain": item.get("domain"),
                "keyword": record["request"]["keyword"],
            })
    return rows


def deduplicate(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge by Google cid, then merge same-name entries within 100 m."""
    raw = raw.assign(key=raw["cid"].fillna(raw["place_id"]))
    groups = raw.groupby("key", sort=False)
    places = groups.first().reset_index()
    places["search_hits"] = groups.size().values
    places["keywords"] = groups["keyword"].agg(lambda keywords: sorted(set(keywords))).values

    places["normalized_name"] = places["name"].map(normalize_name)
    places = places.sort_values("review_count", ascending=False, na_position="last")
    kept, merged = [], 0
    for place in places.itertuples():
        twin = next((other for other in kept if other.normalized_name == place.normalized_name
                     and haversine_km(other.lat, other.lon, place.lat, place.lon) < 0.1), None)
        if twin is None:
            kept.append(place)
        else:
            merged += 1
    if merged:
        print(f"Merged {merged} same-name near-duplicates")
    return places[places["key"].isin([place.key for place in kept])].drop(columns=["keyword"])


def main():
    auth = tuple(require_env("DATAFORSEO_LOGIN", "DATAFORSEO_PASSWORD"))
    config, budget = SETTINGS["place_collection"], Budget()
    polygon = destination_polygon()
    points = grid_points(polygon, config["grid_spacing_km"])
    print(f"{len(points)} grid points x {len(config['keywords'])} keywords = "
          f"{len(points) * len(config['keywords'])} searches")

    rows, saturation, seen = [], [], set()
    for index, (lat, lon) in enumerate(points, start=1):
        for keyword in config["keywords"]:
            items = parse_items(maps_search(keyword, lat, lon, auth, budget))
            rows += items
            seen |= {item["cid"] or item["place_id"] for item in items}
            saturation.append({"search": len(saturation) + 1, "keyword": keyword, "lat": lat, "lon": lon,
                               "returned": len(items), "cumulative_unique": len(seen)})
        print(f"  point {index}/{len(points)}: {len(seen)} unique places so far")

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
    print(f"Spend so far: ${budget.spent():.3f}")


if __name__ == "__main__":
    main()
