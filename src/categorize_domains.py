"""Give every cited domain a category (docs/metrics.md, "Domain categories").

Sources, in order of precedence:
1. websites of places in the place list (domain owned by exactly one place) -> hotel_website
2. config/domains.yaml, filled by hand for the most-cited domains covering 80% of citations
3. the helper LLM for the remaining long tail, flagged source "llm"

The run stops if a domain inside the 80% head is not yet categorised by hand, and writes
those domains to data/review/domains_to_categorise.csv.

    uv run python -m src.categorize_domains             # main run
    uv run python -m src.categorize_domains --pilot     # code test: prints counts only

Output: data/processed/domains[_pilot].parquet
"""

import argparse
import sys
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from src.common import PROCESSED, ROOT, Budget, load_yaml
from src.llm import structured
from src.match_mentions import own_site_domains

STAGE = "domain_categories"
HAND_COVERAGE = 0.80
LLM_BATCH = 40
Category = Literal["ota", "review_site", "hotel_website", "map", "blog_media", "other"]

INSTRUCTIONS = """Categorise web domains cited by AI assistants answering travelers who ask where to stay in
Kaş, Türkiye. Categories:
- ota: online travel agencies, booking platforms and hotel metasearch (incl. Turkish ones)
- review_site: traveler review platforms
- hotel_website: the own website of a single hotel, pension or rental
- map: map services
- blog_media: travel blogs, guides, magazines, news
- other: anything else
Return every domain exactly as given."""


class DomainCategory(BaseModel):
    domain: str
    category: Category


class DomainCategories(BaseModel):
    domains: list[DomainCategory]


def llm_categories(domains: list[str], budget: Budget) -> dict[str, str]:
    result = {}
    for start in range(0, len(domains), LLM_BATCH):
        batch = domains[start:start + LLM_BATCH]
        answer = structured("helper", DomainCategories, [("system", INSTRUCTIONS), ("human", "\n".join(batch))],
                            STAGE, budget)
        result.update({item.domain: item.category for item in answer.domains if item.domain in batch})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="code test on pilot data; no values printed")
    args = parser.parse_args()
    suffix = "_pilot" if args.pilot else ""
    budget = Budget()
    citations = pd.read_parquet(PROCESSED / f"citations{suffix}.parquet")
    counts = citations.domain.value_counts()
    table = pd.DataFrame({"domain": counts.index, "citations": counts.values})
    table["cumulative_share"] = table.citations.cumsum() / table.citations.sum()
    table["in_head"] = table.cumulative_share.shift(fill_value=0) < HAND_COVERAGE

    hand = (load_yaml("domains.yaml") or {}).get("categories") or {}
    place_sites = own_site_domains(pd.read_parquet(PROCESSED / "places.parquet"))
    table["category"] = table.domain.map(lambda d: "hotel_website" if d in place_sites else hand.get(d))
    table["source"] = table.domain.map(lambda d: "place_list" if d in place_sites else ("hand" if d in hand else None))

    missing_head = table[table.in_head & table.category.isna()]
    if len(missing_head) and not args.pilot:
        out = ROOT / "data" / "review" / "domains_to_categorise.csv"
        missing_head[["domain", "citations", "cumulative_share"]].to_csv(out, index=False)
        sys.exit(f"{len(missing_head)} domains in the 80% head need a hand category in config/domains.yaml "
                 f"(listed in {out}).")

    tail = table[table.category.isna()].domain.tolist()
    labels = llm_categories(tail, budget)
    table.loc[table.category.isna(), "source"] = "llm"
    table["category"] = table.category.fillna(table.domain.map(labels)).fillna("other")
    table.to_parquet(PROCESSED / f"domains{suffix}.parquet", index=False)

    print(f"Domains: {len(table)} ({int(table.in_head.sum())} in the 80% head)")
    print("Categorised by:", table.source.value_counts().to_dict())
    if not args.pilot:
        print(table.groupby("category").citations.sum().sort_values(ascending=False).to_string())
    print(f"Spend so far: ${budget.spent():.2f}")


if __name__ == "__main__":
    main()
