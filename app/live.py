"""Run one traveler prompt through Steps 3-4 for the live-query box.

Uses the same code as the pipeline: query_engines.query (cached in data/raw/engines/live,
never part of the metrics), match_mentions.match_answer and apply_corrections. Unmatched
names are classified only from earlier lookups; the live box makes no Serper calls.
"""

from types import SimpleNamespace
from urllib.parse import urlsplit

import pandas as pd

from src.common import SETTINGS, Budget, request_hash
from src.match_mentions import apply_corrections, core_name, match_answer, place_list_text
from src.query_engines import query
from unidecode import unidecode

LIVE_TIMEOUT_SECONDS = 120


def run(text: str, engine: str, places: pd.DataFrame, unmatched: pd.DataFrame, domains: pd.DataFrame) -> dict:
    """Returns answer text, matched places, unmatched names and cited domains. Raises on failure."""
    config = SETTINGS["engines"][engine]
    request = {"engine": engine, "model": config["model"], "init": config["init"], "tools": config["tools"],
               "prompt_id": "live", "text": text.strip(), "repeat": 1}
    budget = Budget()
    record = query(request, budget, live=True, retries=0, timeout=LIVE_TIMEOUT_SECONDS)
    if record.get("error"):
        raise RuntimeError(record["error"])
    if record.get("extraction_error"):
        raise RuntimeError(f"The answer arrived but could not be read: {record['extraction_error']}")
    extracted = record["extracted"]
    answer_id = f"live_{request_hash(request)}"

    grid = places[places.source == "google_maps_grid"] if "source" in places else places
    cores = {p.place_uid: core_name(p.name) or unidecode(p.name).lower() for p in grid.itertuples()}
    rows = match_answer(SimpleNamespace(answer_id=answer_id, text=extracted["text"]), place_list_text(grid),
                        set(grid.place_uid), cores, set(grid[grid.is_accommodation].place_uid), budget)
    citations = pd.DataFrame([{**cite, "answer_id": answer_id,
                               "domain": urlsplit(cite["url"]).netloc.removeprefix("www.") if cite["url"]
                               else (cite["title"] or "").lower()} for cite in extracted["citations"]])
    mentions = pd.DataFrame(rows)
    if len(mentions):
        mentions = apply_corrections(mentions, pd.DataFrame([{"answer_id": answer_id, "text": extracted["text"]}]),
                                     citations if len(citations) else pd.DataFrame(columns=["answer_id", "start",
                                                                                            "end", "domain"]),
                                     places)
    lookup = unmatched.set_index("core_name").category if len(unmatched) else pd.Series(dtype=str)
    named = places.set_index("place_uid")
    matched = mentions[mentions.place_uid.notna()] if len(mentions) else mentions
    unmatched_rows = mentions[mentions.place_uid.isna()] if len(mentions) else mentions
    category = domains.set_index("domain").category
    return {
        "text": extracted["text"], "model": record.get("model_id"), "cost": record.get("cost_usd"),
        "from_cache": record.get("fetched_at"),
        "matched": pd.DataFrame([{"Order": m.order, "Place": named.at[m.place_uid, "name"], "Label": m.label,
                                  "In Kaş list": bool(named.at[m.place_uid, "in_destination"]),
                                  "Named as": m.name_in_answer} for m in matched.itertuples()]),
        "unmatched": pd.DataFrame([{"Named as": m.name_in_answer, "Label": m.label,
                                    "Known from earlier lookups": lookup.get(core_name(m.name_in_answer),
                                                                             "not checked (live)")}
                                   for m in unmatched_rows.itertuples()]),
        "domains": (citations.assign(Category=citations.domain.map(category).fillna("uncategorised"))
                    .groupby(["domain", "Category"]).size().rename("Citations").reset_index()
                    .sort_values("Citations", ascending=False) if len(citations) else citations),
    }
