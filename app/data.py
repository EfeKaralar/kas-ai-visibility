"""Cached loaders for the precomputed tables. The pages read only data/processed, so they
work without internet; only the live-query page calls external APIs."""

import json

import pandas as pd
import streamlit as st

from src.common import PROCESSED, ROOT, SETTINGS

METRICS = PROCESSED / "metrics"
ENGINES = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}
ENGINE_COLORS = {"OpenAI": "#2a78d6", "Gemini": "#eb6834", "Anthropic": "#1baf7a"}
LANGUAGES = {"en": "English", "tr": "Turkish", "de": "German", "ru": "Russian"}
PERSONAS = {"couple_quiet": "Couple seeking quiet", "family_kids": "Family with kids", "diver": "Diver",
            "remote_worker": "Remote worker", "older_couple": "Retired couple", "pet_owner": "Pet owner",
            "honeymooners": "Honeymooners", "road_trippers": "Road-trippers"}
CATEGORY_NAMES = {"ota": "OTA", "review_site": "Review site", "own_website": "Hotel's own website",
                  "hotel_website": "A hotel's website", "map": "Map", "blog_media": "Blog / media",
                  "other": "Other", "no_source": "No source attached"}
CATEGORY_COLORS = {"OTA": "#2a78d6", "Review site": "#eb6834", "Hotel's own website": "#1baf7a",
                   "A hotel's website": "#1baf7a", "Map": "#eda100", "Blog / media": "#e87ba4",
                   "Other": "#bdbcb6", "No source attached": "#e4e3df"}
TARGET_NAMES = {"destination": "Kaş place in our list", "outside_destination": "Outside Kaş",
                "listed_outside_destination": "Outside Kaş", "not_in_place_list": "In Kaş, not in our list",
                "permanently_closed": "Permanently closed", "osm_only": "Only in OpenStreetMap",
                "unverifiable": "Unverifiable", "not_accommodation": "Not accommodation",
                "listed_not_accommodation": "Not accommodation", "in_list_missed_by_matcher": "Matcher miss",
                "unclassified": "Generic name only", "not_looked_up": "Not looked up (named once)",
                "not_on_google_maps": "Real, not on Google Maps"}


@st.cache_data
def metric(name: str) -> pd.DataFrame:
    return pd.read_parquet(METRICS / f"{name}.parquet")


@st.cache_data
def processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@st.cache_data
def recommendations() -> pd.DataFrame:
    recs = metric("recommendations").copy()
    places = processed("places").set_index("place_uid")
    recs["place"] = recs.place_uid.map(places.name).fillna(recs.name_in_answer)
    recs["Engine"] = recs.engine.map(ENGINES)
    recs["Language"] = recs.language.map(LANGUAGES)
    recs["Persona"] = recs.persona.map(PERSONAS)
    recs["Where"] = recs.target.map(TARGET_NAMES).fillna(recs.target)
    return recs


def spend() -> tuple[float, float]:
    ledger = ROOT / SETTINGS["budget"]["ledger_file"]
    spent = sum(json.loads(line)["usd"] for line in ledger.read_text().splitlines() if line) if ledger.exists() else 0
    return spent, SETTINGS["budget"]["hard_cap_usd"]


def doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text()
