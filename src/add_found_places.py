"""Add real Kaş places found only through the answers to the place list.

The grid search missed some places that are on Google Maps (paging blind spot, see
docs/limitations.md). The unmatched-name lookup found them: category not_in_place_list,
inside the polygon, accommodation. This step appends them to data/processed/places.parquet
with source "answer_lookup", using the Google data already in the Serper cache (no credits).
Re-running is safe: places already in the list are skipped.

    uv run python -m src.add_found_places

Then re-run src.classify_unmatched (they become matcher corrections) and src.compute_metrics.
"""

import glob
import json

import pandas as pd

from src.collect_places import normalize_name, parse_places, property_name
from src.common import PROCESSED, RAW, SETTINGS, km_from_center


def google_records() -> dict[str, dict]:
    """cid -> parsed Google Maps item, from every cached Serper response."""
    records = {}
    for path in glob.glob(str(RAW / "serper" / "maps" / "*.json")):
        for item in parse_places(json.loads(open(path).read())):
            records.setdefault(item["cid"], item)
    return records


def main():
    places = pd.read_parquet(PROCESSED / "places.parquet")
    if "source" not in places:
        places["source"] = "google_maps_grid"
    unmatched = pd.read_parquet(PROCESSED / "unmatched_places.parquet")
    found = unmatched[(unmatched.category == "not_in_place_list") & unmatched.cid.notna()
                      & unmatched.in_destination.fillna(False).astype(bool)]
    found = found[~found.cid.isin(places.cid)].drop_duplicates("cid")
    records = google_records()
    accommodation = set(SETTINGS["place_collection"]["accommodation_categories"])

    next_number = int(places.place_uid.str[1:].astype(int).max()) + 1
    rows = []
    for number, row in enumerate(found.itertuples(), start=next_number):
        item = records[row.cid]
        name = property_name(item["name"])     # "X - Double Room with Sea View" -> "X"
        rows.append({**{k: item.get(k) for k in ("cid", "place_id", "category", "categories", "address",
                                                 "lat", "lon", "rating", "review_count", "website")},
                     "name": name, "keywords": [], "search_hits": 0, "listing_name": item["name"],
                     "normalized_name": normalize_name(name), "unit_listings": 0,
                     "is_accommodation": bool(accommodation & {item["category"], *item["categories"]}),
                     "in_destination": True, "km_from_center": round(km_from_center(item["lat"], item["lon"]), 2),
                     "place_uid": "P" + str(number).zfill(4), "source": "answer_lookup",
                     "key": item["cid"]})
    if rows:
        places = pd.concat([places, pd.DataFrame(rows)], ignore_index=True)
        places.to_parquet(PROCESSED / "places.parquet", index=False)
    added = pd.DataFrame(rows)
    print(f"Added {len(rows)} places found through answers; place list now {len(places)} "
          f"({int((places.in_destination & places.is_accommodation).sum())} in the destination)")
    if len(added):
        print(added[["place_uid", "name", "category", "review_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
