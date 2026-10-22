"""Find the places each answer names and match them to the place list.

One helper-LLM call per answer reads the answer together with the full place list and
returns every accommodation the answer names, its match in the list (or none) and whether
it is recommended, named negatively or only incidentally. A fuzzy name match runs alongside
as a cross-check; disagreements are flagged for the hand-check.

    uv run python -m src.match_mentions --pilot     # pilot answers
    uv run python -m src.match_mentions             # main run answers

Mentions the LLM leaves unmatched are then matched by hand-verified aliases
(config/manual_verifications.yaml) or by a nearby citation of a place's own website.

Outputs: data/processed/mentions[_pilot].parquet (one row per place per answer, with its
position in the answer and how it was matched); unmatched names have place_uid empty.
"""

import argparse
import re
from urllib.parse import urlsplit
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from unidecode import unidecode

from src.common import PROCESSED, Budget, load_yaml
from src.llm import structured

STAGE = "mention_matching"
GENERIC_WORDS = re.compile(
    r"\b(hotel|otel|oteli|pansiyon|pension|pansyon|apart|aparts|apartment|apartments|kas|villa|villas|suites?|suit|"
    r"boutique|butik|the|and|club|resort|deluxe|luxury|hostel|motel|adults?|only|spa|restaurant|ve|exclusive)\b")
SUFFIX = re.compile(r"\s*(\(.*?\)|[–—|].*|\s-\s.*|\d+\+)")
LABEL_RANK = {"recommended": 0, "negative": 1, "incidental": 2}

INSTRUCTIONS = """You read an AI assistant's answer to a traveler looking for somewhere to stay in Kaş,
Türkiye, and list every specific place to stay that the answer names: hotels, pensions, apart-hotels,
villas, guest houses, campsites. Ignore restaurants, beaches, beach clubs, dive centres, websites and
areas or neighbourhoods (e.g. "the peninsula", "old town").

For each place, give:
- name_in_answer: the name exactly as written in the answer, character for character (keep the
  answer's language and script), so it can be found in the text.
- place_uid: the id of the same property in the PLACE LIST below, or null if it is not in the list.
  Match despite spelling, transliteration (e.g. Cyrillic), word order, or extra words such as
  "Hotel", "Otel", "Boutique", "Kaş". Only match when you are confident it is the same property.
- label: "recommended" if the answer offers it as an option for the traveler's request, including
  in a plain list of options; "negative" if it is named as something to avoid or that does not fit;
  "incidental" if it is only a landmark, comparison or passing reference.
List each place once, in the order it first appears.

PLACE LIST (id | name | category | where):
{places}"""


class Mention(BaseModel):
    name_in_answer: str
    place_uid: str | None = Field(description="id from the place list, or null")
    label: Literal["recommended", "negative", "incidental"]


class AnswerMentions(BaseModel):
    mentions: list[Mention]


def core_name(name: str) -> str:
    """Distinctive part of a name: no "(Adults Only 14+)"-style suffixes or generic words."""
    name = SUFFIX.sub(" ", unidecode(name or "").lower())
    return " ".join(GENERIC_WORDS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", name)).split())


def place_list_text(places: pd.DataFrame) -> str:
    def where(place):
        if not place.is_accommodation:
            return "not accommodation"
        return "in Kaş" if place.in_destination else f"outside Kaş ({place.km_from_center} km)"
    return "\n".join(f"{p.place_uid} | {p.name} | {p.category} | {where(p)}" for p in places.itertuples())


def find_position(text: str, name: str) -> int | None:
    index = text.lower().find(name.lower())
    return index if index >= 0 else None


def fuzzy_best(name: str, cores: dict[str, str]) -> tuple[str | None, float]:
    query = core_name(name)
    if not query:
        return None, 0.0
    best = process.extractOne(query, cores, scorer=fuzz.token_sort_ratio)
    return (best[2], best[1]) if best else (None, 0.0)


def match_answer(answer, places_text: str, uids: set, cores: dict, lodging: set, budget: Budget) -> list[dict]:
    result = structured("helper", AnswerMentions,
                        [("system", INSTRUCTIONS.format(places=places_text)), ("human", answer.text)],
                        STAGE, budget)
    rows = {}
    for mention in result.mentions:
        uid = mention.place_uid if mention.place_uid in uids else None
        fuzzy_uid, fuzzy_score = fuzzy_best(mention.name_in_answer, cores)
        llm_vs_fuzzy = fuzz.token_sort_ratio(core_name(mention.name_in_answer), cores[uid]) if uid else None
        row = {"answer_id": answer.answer_id, "name_in_answer": mention.name_in_answer, "place_uid": uid,
               "label": mention.label, "position": find_position(answer.text, mention.name_in_answer),
               "llm_uid_unknown": mention.place_uid is not None and uid is None,
               "fuzzy_uid": fuzzy_uid, "fuzzy_score": round(fuzzy_score, 1),
               # flag disagreements with the fuzzy match, and matches to places that are not accommodation
               "needs_check": bool((uid and llm_vs_fuzzy < 60) or (not uid and fuzzy_score >= 90)
                                   or (uid and uid not in lodging))}
        key = uid or core_name(mention.name_in_answer) or mention.name_in_answer
        # one row per place per answer; the strongest label wins (recommended > negative > incidental)
        if key not in rows or LABEL_RANK[row["label"]] < LABEL_RANK[rows[key]["label"]]:
            rows[key] = row
    ordered = sorted(rows.values(), key=lambda r: (r["position"] is None, r["position"] or 0))
    for order, row in enumerate(ordered, start=1):
        row["order"] = order
    return ordered


def own_site_domains(places: pd.DataFrame) -> dict[str, str]:
    """Website domain -> place_uid, only for domains that belong to exactly one place
    (a website field pointing at booking.com or instagram.com says nothing about identity)."""
    domains = places.website.dropna().map(lambda url: urlsplit(url).netloc.lower().removeprefix("www."))
    owners = places.loc[domains.index, "place_uid"].groupby(domains.values).agg(list)
    return {domain: uids[0] for domain, uids in owners.items() if domain and len(uids) == 1}


def paragraph_bounds(text: str, position: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return start, len(text) if end < 0 else end


def apply_corrections(mentions: pd.DataFrame, answers: pd.DataFrame, citations: pd.DataFrame,
                      places: pd.DataFrame) -> pd.DataFrame:
    """Match mentions the LLM left unmatched, using (1) hand-verified aliases from
    config/manual_verifications.yaml and (2) a citation in the same paragraph whose domain is
    the website of exactly one place in the list and contains the mentioned name. Website
    matches are flagged for checking."""
    verified = load_yaml("manual_verifications.yaml") or {}
    cid_to_uid = dict(zip(places.cid, places.place_uid))
    aliases = {core: cid_to_uid[str(facts["google_cid"])] for core, facts in verified.items()
               if facts["status"] == "same_as" and str(facts["google_cid"]) in cid_to_uid}
    site_owner = own_site_domains(places)
    texts = answers.set_index("answer_id").text
    mentions = mentions.assign(matched_by=mentions.place_uid.notna().map({True: "llm", False: None}))
    for index, mention in mentions[mentions.place_uid.isna()].iterrows():
        core = core_name(mention.name_in_answer)
        if core in aliases:
            mentions.loc[index, ["place_uid", "matched_by"]] = [aliases[core], "manual"]
            continue
        if mention.position is None or pd.isna(mention.position):
            continue
        start, end = paragraph_bounds(texts[mention.answer_id], int(mention.position))
        nearby = citations[(citations.answer_id == mention.answer_id) & (citations.start < end)
                           & (citations.end > start)]
        # the domain must also contain the name as written (e.g. "Sea Garden" -> seagardenkas.com);
        # a paragraph often cites several hotels' sites, so proximity alone matches the wrong one
        squashed = re.sub(r"[^a-z0-9]", "", core)
        owners = {site_owner[domain] for domain in nearby.domain
                  if domain in site_owner and len(squashed) >= 5 and squashed in re.sub(r"[^a-z0-9]", "", domain)}
        if len(owners) == 1:
            mentions.loc[index, ["place_uid", "matched_by", "needs_check"]] = [owners.pop(), "website", True]
    # a correction can make two rows in one answer refer to the same place: keep the one with
    # the strongest label (then the earliest position); unmatched rows are left as they are
    matched = (mentions[mentions.place_uid.notna()].assign(rank=lambda df: df.label.map(LABEL_RANK))
               .sort_values(["rank", "position"]).drop_duplicates(["answer_id", "place_uid"])
               .drop(columns="rank"))
    mentions = pd.concat([matched, mentions[mentions.place_uid.isna()]]).sort_values(["answer_id", "position"])
    mentions["order"] = mentions.groupby("answer_id").cumcount() + 1
    return mentions.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    suffix = "_pilot" if args.pilot else ""
    budget = Budget()
    answers = pd.read_parquet(PROCESSED / f"answers{suffix}.parquet")
    answers = answers[answers.error.isna() & answers.text.notna()]
    places = pd.read_parquet(PROCESSED / "places.parquet")
    places_text = place_list_text(places)
    uids = set(places.place_uid)
    lodging = set(places[places.is_accommodation].place_uid)
    # names made only of generic words (e.g. "Kaş Otel") keep their full name for comparison
    cores = {p.place_uid: core_name(p.name) or unidecode(p.name).lower() for p in places.itertuples()}

    rows = []
    for done, answer in enumerate(answers.itertuples(), start=1):
        rows += match_answer(answer, places_text, uids, cores, lodging, budget)
        if done % 20 == 0 or done == len(answers):
            print(f"  {done}/{len(answers)} answers matched, spent ${budget.spent():.2f}")

    citations = pd.read_parquet(PROCESSED / f"citations{suffix}.parquet")
    mentions = apply_corrections(pd.DataFrame(rows), answers, citations, places).merge(
        places[["place_uid", "name", "in_destination", "is_accommodation"]], on="place_uid", how="left")
    mentions.to_parquet(PROCESSED / f"mentions{suffix}.parquet", index=False)

    matched = mentions.place_uid.notna()
    print(f"\nMentions: {len(mentions)} in {mentions.answer_id.nunique()} answers "
          f"({matched.mean():.0%} matched to the list)")
    print(mentions.label.value_counts().to_string())
    print(f"Recommended per answer (median): "
          f"{mentions[mentions.label == 'recommended'].groupby('answer_id').size().median()}")
    print("Matched by:", mentions.matched_by.value_counts().to_dict())
    print(f"Flagged for checking: {mentions.needs_check.sum()}")
    print(f"Name not found verbatim in the answer: {mentions.position.isna().sum()}")
    print(f"Spend so far: ${budget.spent():.2f}")


if __name__ == "__main__":
    main()
