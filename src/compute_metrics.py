"""Compute the metrics defined in docs/metrics.md as tidy tables.

    uv run python -m src.compute_metrics             # main run -> data/processed/metrics/
    uv run python -m src.compute_metrics --pilot     # code test: checks and row counts only

Inputs: answers, citations, mentions, unmatched_places, domains (all from earlier steps),
places and prompts. Every per-place denominator is destination_places(); sanity checks
from docs/metrics.md run first and stop the script if they fail.
"""

import argparse
import itertools

import pandas as pd

from src.common import PROCESSED, check_population, destination_places
from src.categorize_domains import own_sites
from src.match_mentions import core_name

TOP_N = 5


# --- loading -------------------------------------------------------------------

def load(suffix: str) -> dict:
    data = {name: pd.read_parquet(PROCESSED / f"{name}{suffix}.parquet")
            for name in ("answers", "citations", "mentions", "unmatched_places", "domains")}
    data["places"] = pd.read_parquet(PROCESSED / "places.parquet")
    data["destination"] = destination_places()
    prompts = pd.read_parquet(PROCESSED / "prompts.parquet")[["prompt_id", "base_id", "persona", "language"]]
    answers = data["answers"].merge(prompts, on="prompt_id", how="left")
    data["failed"] = answers[answers.error.notna()]
    data["answers"] = answers[answers.error.isna()]
    return data


def recommendations(data: dict) -> pd.DataFrame:
    """Recommended mentions with matcher corrections, answer context and a target category (V7)."""
    mentions = data["mentions"].copy()
    unmatched = data["unmatched_places"].set_index("core_name")
    missed = unmatched[unmatched.category == "in_list_missed_by_matcher"].place_uid
    cores = mentions.name_in_answer.map(core_name)
    fix = mentions.place_uid.isna() & cores.isin(missed.index)
    mentions.loc[fix, "place_uid"] = cores[fix].map(missed)
    mentions.loc[fix, "matched_by"] = "lookup"

    places = data["places"].set_index("place_uid")
    destination = set(data["destination"].place_uid)

    def target(row):
        if pd.notna(row.place_uid):
            if row.place_uid in destination:
                return "destination"
            return "listed_not_accommodation" if not places.at[row.place_uid, "is_accommodation"] \
                else "listed_outside_destination"
        core = core_name(row.name_in_answer)
        return unmatched.category.get(core, "unclassified") if core else "unclassified"

    mentions["target"] = mentions.apply(target, axis=1)
    recs = mentions[mentions.label == "recommended"]
    context = data["answers"][["answer_id", "engine", "language", "persona", "base_id", "prompt_id", "repeat"]]
    return recs.merge(context, on="answer_id", how="inner")


# --- sanity checks -------------------------------------------------------------

def sanity_checks(data: dict, recs: pd.DataFrame, expected_answers: int) -> list[str]:
    report = []
    total = len(data["answers"]) + len(data["failed"])
    assert total == expected_answers, f"expected {expected_answers} answers, found {total}"
    report.append(f"answers: {len(data['answers'])} ok, {len(data['failed'])} failed "
                  f"{data['failed'].groupby('engine').size().to_dict()}")
    unknown = set(recs.place_uid.dropna()) - set(data["places"].place_uid)
    assert not unknown, f"recommendations point to unknown places: {unknown}"
    duplicates = data["mentions"].dropna(subset=["place_uid"]).duplicated(["answer_id", "place_uid"]).sum()
    assert duplicates == 0, f"{duplicates} duplicate answer x place rows"
    lengths = data["answers"].set_index("answer_id").text.str.len()
    cites = data["citations"][data["citations"].answer_id.isin(lengths.index)]
    out_of_range = (cites.start < 0) | (cites.end > cites.answer_id.map(lengths)) | (cites.start >= cites.end)
    assert not out_of_range.any(), f"{out_of_range.sum()} citation spans outside their answer"
    report.append(f"recommendations: {len(recs)}; place ids valid; no duplicates; citation spans valid")
    return report


# --- visibility metrics ----------------------------------------------------------

def per_place(recs: pd.DataFrame, destination: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V1/V2 per place (all destination places, zeros included) and the long breakdown."""
    in_dest = recs[recs.target == "destination"]
    summary = destination[["place_uid", "name", "category", "rating", "review_count", "website",
                           "km_from_center"]].copy()
    summary["recommendations"] = summary.place_uid.map(in_dest.groupby("place_uid").size()).fillna(0).astype(int)
    summary["distinct_base_prompts"] = (summary.place_uid.map(in_dest.groupby("place_uid").base_id.nunique())
                                        .fillna(0).astype(int))
    check_population(summary.place_uid, "per_place")
    breakdown = (in_dest.groupby(["place_uid", "engine", "language", "persona"]).size()
                 .rename("recommendations").reset_index())
    return summary, breakdown


def coverage(recs: pd.DataFrame, destination: pd.DataFrame) -> pd.DataFrame:
    """V3: share of destination places recommended at least once, overall and per engine/language."""
    in_dest = recs[recs.target == "destination"]
    total = len(destination)
    rows = [{"scope": "overall", "group": "all", "places_recommended": in_dest.place_uid.nunique()}]
    for scope in ("engine", "language"):
        for group, part in in_dest.groupby(scope):
            rows.append({"scope": scope, "group": group, "places_recommended": part.place_uid.nunique()})
    table = pd.DataFrame(rows)
    table["destination_places"] = total
    table["coverage"] = table.places_recommended / total
    return table


def concentration(recs: pd.DataFrame) -> pd.DataFrame:
    """V4: share of destination recommendations going to the top 5 places, overall and per engine."""
    in_dest = recs[recs.target == "destination"]
    rows = []
    for scope, group, part in [("overall", "all", in_dest)] + [("engine", g, p) for g, p in in_dest.groupby("engine")]:
        counts = part.place_uid.value_counts()
        rows.append({"scope": scope, "group": group, "recommendations": int(counts.sum()),
                     "top_n": TOP_N, "top_n_share": counts.head(TOP_N).sum() / counts.sum() if len(counts) else None,
                     "top_n_places": list(counts.head(TOP_N).index)})
    return pd.DataFrame(rows)


def popularity(summary: pd.DataFrame) -> pd.DataFrame:
    """V5: Spearman correlation of recommendations with review count (descriptive)."""
    known = summary.dropna(subset=["review_count"])
    # Spearman's rho = Pearson correlation of the ranks (average ranks for ties)
    rho = known.review_count.rank().corr(known.recommendations.rank())
    return pd.DataFrame([{"places": len(known), "spearman_rho": rho}])


def no_recommendation(data: dict, recs: pd.DataFrame) -> pd.DataFrame:
    """V6: share of answers without any recommendation, overall and per engine/language."""
    answers = data["answers"].assign(has_recommendation=lambda df: df.answer_id.isin(recs.answer_id))
    rows = [{"scope": "overall", "group": "all", "answers": len(answers),
             "no_recommendation_rate": 1 - answers.has_recommendation.mean()}]
    for scope in ("engine", "language"):
        for group, part in answers.groupby(scope):
            rows.append({"scope": scope, "group": group, "answers": len(part),
                         "no_recommendation_rate": 1 - part.has_recommendation.mean()})
    return pd.DataFrame(rows)


def targets(recs: pd.DataFrame) -> pd.DataFrame:
    """V7: where recommendations point, overall and per engine."""
    overall = recs.target.value_counts().rename("recommendations").reset_index().assign(engine="all")
    by_engine = recs.groupby(["engine", "target"]).size().rename("recommendations").reset_index()
    table = pd.concat([overall, by_engine], ignore_index=True)
    table["share"] = table.recommendations / table.groupby("engine").recommendations.transform("sum")
    return table


def repeat_stability(recs: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """V8: mean Jaccard overlap of recommended place sets between repeats of the same prompt x engine."""
    sets = recs.dropna(subset=["place_uid"]).groupby("answer_id").place_uid.agg(set)
    rows = []
    for (prompt_id, engine), group in answers.groupby(["prompt_id", "engine"]):
        place_sets = [sets.get(answer_id, set()) for answer_id in group.answer_id]
        for first, second in itertools.combinations(place_sets, 2):
            union = first | second
            rows.append({"engine": engine, "prompt_id": prompt_id,
                         "jaccard": len(first & second) / len(union) if union else None})
    pairs = pd.DataFrame(rows, columns=["engine", "prompt_id", "jaccard"])
    return pairs.groupby("engine").jaccard.agg(["mean", "count"]).rename(columns={"count": "pairs"}).reset_index()


# --- source metrics --------------------------------------------------------------

def segments(answers: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    """Each recommendation's segment: from its line to the line where the next place is first named."""
    texts = answers.set_index("answer_id").text
    rows = []
    positioned = recs.dropna(subset=["position"])
    for answer_id, group in positioned.groupby("answer_id"):
        text = texts[answer_id]
        line_start = {i: text.rfind("\n", 0, int(p)) + 1 for i, p in zip(group.index, group.position)}
        starts = sorted(set(line_start.values()))
        for index, start in line_start.items():
            later = [s for s in starts if s > start]
            rows.append({"rec_index": index, "answer_id": answer_id, "segment_start": start,
                         "segment_end": later[0] if later else len(text)})
    return pd.DataFrame(rows)


def citation_links(data: dict, recs: pd.DataFrame) -> pd.DataFrame:
    """One row per (recommendation, supporting domain), categorised per docs/metrics.md."""
    domains = data["domains"].set_index("domain").category
    site_owner = own_sites(data["places"])
    bounds = segments(data["answers"], recs).set_index("rec_index")
    cites = data["citations"]
    rows = []
    for index, rec in recs.iterrows():
        if index not in bounds.index:
            continue
        span = bounds.loc[index]
        support = cites[(cites.answer_id == rec.answer_id) & (cites.start < span.segment_end)
                        & (cites.end > span.segment_start)]
        base = {"rec_index": index, "answer_id": rec.answer_id, "place_uid": rec.place_uid, "target": rec.target,
                "engine": rec.engine, "language": rec.language}
        if support.empty:
            rows.append({**base, "domain": None, "category": "no_source"})
        for domain in support.domain.dropna().unique():
            if pd.notna(rec.place_uid) and site_owner.get(domain) == rec.place_uid:
                category = "own_website"
            else:
                category = domains.get(domain, "other")
                category = "other" if category == "hotel_website" else category
            rows.append({**base, "domain": domain, "category": category})
    return pd.DataFrame(rows)


def sources_per_answer(data: dict) -> pd.DataFrame:
    """S1: share of answers citing each category at least once, overall and per engine/language."""
    domains = data["domains"].set_index("domain").category
    cited = data["citations"].assign(category=lambda df: df.domain.map(domains).fillna("other"))
    answers = data["answers"][["answer_id", "engine", "language"]]
    has = (cited.groupby(["answer_id", "category"]).size().unstack(fill_value=0) > 0)
    has = answers.merge(has, left_on="answer_id", right_index=True, how="left").fillna(False)
    categories = [c for c in has.columns if c not in ("answer_id", "engine", "language")]
    rows = []
    for scope, keys in (("overall", None), ("engine", "engine"), ("language", "language")):
        groups = [("all", has)] if keys is None else list(has.groupby(keys))
        for group, part in groups:
            for category in categories:
                rows.append({"scope": scope, "group": group, "category": category,
                             "share_of_answers": part[category].astype(bool).mean()})
    return pd.DataFrame(rows)


def sources_per_recommendation(links: pd.DataFrame) -> pd.DataFrame:
    """S2: category shares over (recommendation, domain) pairs, plus no-source share of recommendations."""
    rows = []
    for scope, keys in (("overall", None), ("engine", "engine"), ("language", "language")):
        groups = [("all", links)] if keys is None else list(links.groupby(keys))
        for group, part in groups:
            pairs = part[part.category != "no_source"]
            for category, count in pairs.category.value_counts().items():
                rows.append({"scope": scope, "group": group, "category": category, "pairs": int(count),
                             "share_of_pairs": count / len(pairs)})
            recs_total = part.rec_index.nunique()
            no_source = part[part.category == "no_source"].rec_index.nunique()
            rows.append({"scope": scope, "group": group, "category": "no_source", "pairs": no_source,
                         "share_of_recommendations": no_source / recs_total if recs_total else None})
    return pd.DataFrame(rows)


def hypothesis(links: pd.DataFrame, destination: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """H: own website vs OTA backing for recommended destination places that have a website."""
    with_site = set(destination.dropna(subset=["website"]).place_uid)
    eligible = links[(links.target == "destination") & links.place_uid.isin(with_site)]
    per_rec = eligible.groupby(["rec_index", "engine"]).category.agg(set).reset_index()
    per_rec["own"] = per_rec.category.map(lambda c: "own_website" in c)
    per_rec["ota"] = per_rec.category.map(lambda c: "ota" in c)
    per_rec["no_source"] = per_rec.category.map(lambda c: c == {"no_source"})

    def shares(part, group):
        return {"group": group, "recommendations": len(part), "own_website": part.own.mean(),
                "ota": part.ota.mean(), "both": (part.own & part.ota).mean(),
                "neither": (~part.own & ~part.ota & ~part.no_source).mean(), "no_source": part.no_source.mean()}

    table = pd.DataFrame([shares(per_rec, "all")] + [shares(p, g) for g, p in per_rec.groupby("engine")])
    overall = table.iloc[0]
    engines = table.iloc[1:]
    if overall.own_website >= overall.ota:
        verdict = "not supported"
    elif (engines.ota > engines.own_website).all():
        verdict = "supported"
    else:
        verdict = "mixed"
    return table, verdict


def robustness(links: pd.DataFrame, data: dict) -> tuple[pd.DataFrame, str]:
    """docs/metrics.md robustness check: drop citations of LLM-categorised domains and recompute
    the source shares and the hypothesis. Own-website links never depend on the LLM."""
    llm_domains = set(data["domains"][data["domains"].source == "llm"].domain)
    kept = links[~links.domain.isin(llm_domains)]
    table, verdict = hypothesis(kept, data["destination"])
    shares = sources_per_recommendation(kept).query("scope == 'overall' and category != 'no_source'")
    table = table.assign(verdict=verdict, dropped_links=int(links.domain.isin(llm_domains).sum()))
    return pd.concat([table.assign(table_kind="hypothesis"),
                      shares.assign(table_kind="sources_overall")], ignore_index=True), verdict


# --- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="code test on pilot data; no values printed")
    args = parser.parse_args()
    suffix = "_pilot" if args.pilot else ""
    data = load(suffix)
    recs = recommendations(data)
    for line in sanity_checks(data, recs, 48 if args.pilot else 720):
        print("check:", line)

    summary, breakdown = per_place(recs, data["destination"])
    links = citation_links(data, recs)
    hypothesis_table, verdict = hypothesis(links, data["destination"])
    robust_table, robust_verdict = robustness(links, data)
    tables = {
        "place_summary": summary, "place_breakdown": breakdown,
        "coverage": coverage(recs, data["destination"]), "concentration": concentration(recs),
        "popularity": popularity(summary), "no_recommendation": no_recommendation(data, recs),
        "recommendation_targets": targets(recs), "repeat_stability": repeat_stability(recs, data["answers"]),
        "citation_links": links, "sources_per_answer": sources_per_answer(data),
        "sources_per_recommendation": sources_per_recommendation(links),
        "hypothesis": hypothesis_table.assign(verdict=verdict),
        "robustness_without_llm_domains": robust_table,
    }
    folder = PROCESSED / f"metrics{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(folder / f"{name}.parquet", index=False)
        print(f"  {name}: {len(table)} rows")
    if not args.pilot:
        print(f"\nHypothesis verdict: {verdict} (without LLM-categorised domains: {robust_verdict})")


if __name__ == "__main__":
    main()
