"""Classify places that answers name but that are not in the place list.

Each distinct unmatched name is looked up on Google Maps (Serper, cached) as "<name> Kaş", and if
that finds nothing, once more as "<name>" alone. A result is accepted when its name matches closely,
or, failing that, when the helper LLM judges one of the top results to be the same property
(names in answers are often shortened or reworded). Then:
- found, accommodation, inside the polygon  -> "not_in_place_list" (real, our search missed it)
- found, accommodation, outside the polygon -> "outside_destination"
- found but already in our list             -> "in_list_missed_by_matcher" (matcher error)
- not on Google, but in OpenStreetMap        -> "osm_only" (exists in OSM; possibly closed or unlisted)
- no confident match anywhere                -> "unverifiable" (possibly closed or non-existent)

    uv run python -m src.classify_unmatched --pilot
    uv run python -m src.classify_unmatched

Output: data/processed/unmatched_places[_pilot].parquet, one row per distinct name.
"""

import argparse

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.collect_places import parse_places
from src.common import PROCESSED, SETTINGS, Budget, in_destination, km_from_center
from src.match_mentions import core_name
from src.llm import structured
from src.review_polygon import fetch_osm
from src.serper import Serper

STAGE = "unmatched_lookup"
LOOKUP_ZOOM = "12z"          # wide view: places outside Kaş must be findable too
MIN_NAME_SCORE = 85
CANDIDATES_FOR_LLM = 5

JUDGE_INSTRUCTIONS = """An AI travel answer named a place to stay. Decide which of the Google Maps results,
if any, is the same property. Names may be shortened, translated or reworded (e.g. "Hotel X" vs
"X Butik Otel"). Answer with the result number, or null if none is clearly the same property."""


class SameProperty(BaseModel):
    result_number: int | None = Field(description="1-based number of the matching result, or null")


def best_result(name: str, results: list[dict]) -> tuple[dict | None, float]:
    target = core_name(name)
    scored = [(fuzz.token_sort_ratio(target, core_name(item["name"])), item) for item in results]
    scored = [(score, item) for score, item in scored if core_name(item["name"])]
    if not scored:
        return None, 0.0
    score, item = max(scored, key=lambda pair: pair[0])
    return (item, score) if score >= MIN_NAME_SCORE else (None, score)


def judge_with_llm(name: str, results: list[dict], budget: Budget) -> dict | None:
    candidates = results[:CANDIDATES_FOR_LLM]
    if not candidates:
        return None
    listing = "\n".join(f"{n}. {item['name']} | {item['category']} | {item['address']}"
                        for n, item in enumerate(candidates, start=1))
    verdict = structured("helper", SameProperty,
                         [("system", JUDGE_INSTRUCTIONS), ("human", f"Name in answer: {name}\n\nResults:\n{listing}")],
                         STAGE, budget)
    number = verdict.result_number
    return candidates[number - 1] if number and 1 <= number <= len(candidates) else None


def look_up(name: str, client: Serper, budget: Budget) -> tuple[dict | None, float, str]:
    """Returns (result, fuzzy score, how it was found)."""
    center_lat, center_lon = SETTINGS["destination"]["center"]
    best_score = 0.0
    for query in (f"{name} Kaş", name):
        results = parse_places(client.maps_search(query, center_lat, center_lon, zoom=LOOKUP_ZOOM))
        found, score = best_result(name, results)
        best_score = max(best_score, score)
        if found:
            return found, score, "name"
        if found := judge_with_llm(name, results, budget):
            return found, score, "llm"
    return None, best_score, "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    suffix = "_pilot" if args.pilot else ""
    budget = Budget()
    client = Serper(STAGE, budget)
    accommodation = set(SETTINGS["place_collection"]["accommodation_categories"])
    places = pd.read_parquet(PROCESSED / "places.parquet")
    known_cids = dict(zip(places.cid, places.place_uid))
    osm_places = [place for place in fetch_osm() if core_name(place["name"])]

    mentions = pd.read_parquet(PROCESSED / f"mentions{suffix}.parquet")
    unmatched = mentions[mentions.place_uid.isna()].assign(core=lambda df: df.name_in_answer.map(core_name))
    unmatched = unmatched[unmatched.core != ""]
    groups = unmatched.groupby("core")
    print(f"{len(unmatched)} unmatched mentions, {groups.ngroups} distinct names; "
          f"Serper credits used so far {client.credits_used()}")

    rows = []
    for core, group in groups:
        name = group.name_in_answer.value_counts().index[0]   # most common spelling
        found, score, method = look_up(name, client, budget)
        row = {"core_name": core, "name": name, "spellings": sorted(set(group.name_in_answer)),
               "mentions": len(group), "recommended": int((group.label == "recommended").sum()),
               "answers": group.answer_id.nunique(), "match_score": round(score, 1), "found_by": method}
        if found is None:
            osm = next((place for place in osm_places
                        if fuzz.token_sort_ratio(core, core_name(place["name"])) >= MIN_NAME_SCORE), None)
            row["category"] = "osm_only" if osm else "unverifiable"
            if osm:
                row.update({"found_name": osm["name"], "lat": osm["lat"], "lon": osm["lon"],
                            "km_from_center": round(km_from_center(osm["lat"], osm["lon"]), 2),
                            "in_destination": in_destination(osm["lat"], osm["lon"])})
        else:
            is_lodging = bool(accommodation & {found["category"], *found["categories"]})
            inside = in_destination(found["lat"], found["lon"])
            row.update({"found_name": found["name"], "google_category": found["category"], "cid": found["cid"],
                        "lat": found["lat"], "lon": found["lon"], "review_count": found["review_count"],
                        "km_from_center": round(km_from_center(found["lat"], found["lon"]), 2),
                        "is_accommodation": is_lodging, "in_destination": inside})
            if found["cid"] in known_cids:
                row["category"], row["place_uid"] = "in_list_missed_by_matcher", known_cids[found["cid"]]
            elif not is_lodging:
                row["category"] = "not_accommodation"
            else:
                row["category"] = "not_in_place_list" if inside else "outside_destination"
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_parquet(PROCESSED / f"unmatched_places{suffix}.parquet", index=False)
    print(table.groupby("category").agg(names=("name", "size"), mentions=("mentions", "sum")).to_string())
    print(f"\nSerper credits used: {client.credits_used()}")


if __name__ == "__main__":
    main()
