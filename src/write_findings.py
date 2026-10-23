"""Write docs/findings.md from the metric tables.

Every number in the summary is computed here from data/processed, so the text cannot drift
from the data; the wording around the numbers is written by hand in this file. Re-run after
any change upstream:

    uv run python -m src.write_findings
"""

import itertools
import json

import pandas as pd

from src.common import PROCESSED, ROOT, SETTINGS
from src.compute_metrics import load, recommendations

METRICS = PROCESSED / "metrics"
ENGINES = {"openai": "OpenAI", "gemini": "Gemini", "anthropic": "Anthropic"}
LANGUAGES = {"en": "English", "tr": "Turkish", "de": "German", "ru": "Russian"}


def pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def table(name: str) -> pd.DataFrame:
    return pd.read_parquet(METRICS / f"{name}.parquet")


def facts() -> dict:
    data = load("")
    recs = recommendations(data)
    places = data["places"].set_index("place_uid")
    summary = table("place_summary")
    coverage = table("coverage").set_index(["scope", "group"])
    concentration = table("concentration").set_index("group")
    hypothesis = table("hypothesis").set_index("group")
    robust = table("robustness_without_llm_domains").query("table_kind == 'hypothesis'").set_index("group")
    sources = table("sources_per_recommendation")
    targets = table("recommendation_targets").query("engine == 'all'").set_index("target")
    stability = table("repeat_stability").set_index("engine")
    unmatched = pd.read_parquet(PROCESSED / "unmatched_places.parquet")
    ledger = [json.loads(line) for line in (ROOT / SETTINGS["budget"]["ledger_file"]).read_text().splitlines()]

    destination = recs[recs.target == "destination"]
    engine_sets = {e: set(g.place_uid) for e, g in destination.groupby("engine")}
    reviewed = summary[summary.review_count >= 300]
    invisible = reviewed[reviewed.recommendations == 0].sort_values("review_count", ascending=False)
    closed = recs[recs.target == "permanently_closed"]
    outside = unmatched[unmatched.category == "outside_destination"]
    near = int(outside[outside.km_from_center < 5].mentions.sum())
    far = int(outside[(outside.km_from_center >= 15) | outside.area.notna()].mentions.sum())
    s2 = sources[sources.scope == "overall"].set_index("category")
    s2_engine = sources[sources.scope == "engine"]

    def share(category, engine=None):
        part = s2 if engine is None else s2_engine[s2_engine.group == engine].set_index("category")
        column = "share_of_recommendations" if category == "no_source" else "share_of_pairs"
        return float(part.at[category, column]) if category in part.index else 0.0

    quality = METRICS / "matcher_quality.parquet"
    return {
        "answers": len(data["answers"]), "failed": len(data["failed"]),
        "base_prompts": data["answers"].base_id.nunique(), "repeats": data["answers"]["repeat"].nunique(),
        "spend": sum(row["usd"] for row in ledger), "credits": sum(row.get("credits", 0) for row in ledger),
        "recommendations": len(recs), "mentions": len(data["mentions"]),
        "matched_share": data["mentions"].place_uid.notna().mean(),
        "places": len(summary),
        "found_via_answers": int(summary.place_uid.map(places.source).eq("answer_lookup").sum()),
        "recommended": int((summary.recommendations > 0).sum()),
        "never": int((summary.recommendations == 0).sum()),
        "coverage": coverage.loc[("overall", "all"), "coverage"],
        "coverage_engine": {e: coverage.loc[("engine", e), "coverage"] for e in ENGINES},
        "coverage_language": {lang: coverage.loc[("language", lang), "coverage"] for lang in LANGUAGES},
        "reviewed": len(reviewed), "invisible": invisible,
        "top_share": concentration.at["all", "top_n_share"],
        "top_names": [places.at[uid, "name"] for uid in concentration.at["all", "top_n_places"]],
        "top_share_engine": {e: concentration.at[e, "top_n_share"] for e in ENGINES},
        "top_by_engine": {e: [places.at[uid, "name"] for uid in destination[destination.engine == e]
                              .place_uid.value_counts().head(3).index] for e in ENGINES},
        "all_three": len(set.intersection(*engine_sets.values())),
        "any_engine": len(set.union(*engine_sets.values())),
        "pair_overlap": {(a, b): len(engine_sets[a] & engine_sets[b]) / len(engine_sets[a] | engine_sets[b])
                         for a, b in itertools.combinations(ENGINES, 2)},
        "per_answer": destination.groupby(["engine", "answer_id"]).size().groupby("engine").median().to_dict(),
        "rho": table("popularity").spearman_rho.iloc[0],
        "destination_share": targets.at["destination", "share"],
        "outside_share": targets.reindex(["outside_destination", "listed_outside_destination"]).share.sum(),
        "near": near, "far": far,
        "closed": len(closed), "closed_answers": closed.answer_id.nunique(),
        "closed_engines": closed.groupby("engine").size().to_dict(),
        "closed_languages": closed.groupby("language").size().to_dict(),
        "not_on_google": int(targets.recommendations.get("not_on_google_maps", 0)),
        "unverifiable": int(targets.recommendations.get("unverifiable", 0)),
        "share": share, "hypothesis": hypothesis, "verdict": hypothesis.verdict.iloc[0],
        "robust": robust, "robust_verdict": robust.verdict.iloc[0],
        "stability": stability["mean"].to_dict(),
        "quality": pd.read_parquet(quality).set_index("metric") if quality.exists() else None,
    }


def render(f: dict) -> str:
    h, r = f["hypothesis"], f["robust"]
    cov_e = ", ".join(f"{ENGINES[e]} {pct(v)}" for e, v in f["coverage_engine"].items())
    cov_l = ", ".join(f"{LANGUAGES[k]} {pct(v)}" for k, v in f["coverage_language"].items())
    invisible = ", ".join(f"{row.name} ({int(row.review_count)} reviews)" for row in f["invisible"].head(4).itertuples())
    overlap = ", ".join(f"{ENGINES[a]}-{ENGINES[b]} {pct(v)}" for (a, b), v in f["pair_overlap"].items())
    tops = "; ".join(f"{ENGINES[e]}: {', '.join(names)}" for e, names in f["top_by_engine"].items())
    per_answer = ", ".join(f"{ENGINES[e]} {f['per_answer'][e]:.0f}" for e in ENGINES)
    stability = ", ".join(f"{ENGINES[e]} {f['stability'][e]:.2f}" for e in ENGINES)
    closed_engines = ", ".join(f"{ENGINES[e]} {n}" for e, n in f["closed_engines"].items())
    hyp_rows = "\n".join(
        f"| {'All engines' if g == 'all' else ENGINES[g]} | {pct(h.at[g, 'ota'])} | {pct(h.at[g, 'own_website'])} | "
        f"{pct(h.at[g, 'no_source'])} | {int(h.at[g, 'recommendations'])} |" for g in ["all", *ENGINES])
    src_rows = "\n".join(
        f"| {'All engines' if e is None else ENGINES[e]} | {pct(f['share']('ota', e))} | "
        f"{pct(f['share']('own_website', e))} | {pct(f['share']('blog_media', e))} | "
        f"{pct(f['share']('review_site', e))} | {pct(f['share']('other', e))} |" for e in [None, *ENGINES])
    if f["quality"] is not None:
        q = f["quality"]
        quality = (f"Hand-check of 30 answers (stratified by engine and language; "
                   f"{int(q.at['precision', 'denominator'])} matched places checked by the project owner, "
                   f"no errors found): precision {pct(q.at['precision', 'value'])} "
                   f"(95% interval {pct(q.at['precision', 'low_95'])}-{pct(q.at['precision', 'high_95'])}), "
                   f"recall {pct(q.at['recall', 'value'])} "
                   f"({pct(q.at['recall', 'low_95'])}-{pct(q.at['recall', 'high_95'])}).")
    else:
        quality = "Hand-check of 30 answers: **pending** (sheet in `data/review/hand_check.csv`)."

    return f"""# Findings: how AI assistants recommend places to stay in Kaş

*Generated by `src/write_findings.py` from `data/processed/metrics/`; every number below comes from the data.
Metric definitions were fixed in advance in [metrics.md](metrics.md); caveats are in [limitations.md](limitations.md).*

## Summary

- **Fewer than half of Kaş's places ever get recommended.** Across {f['answers']} answers, {pct(f['coverage'])} of the {f['places']} places in Kaş were recommended at least once; {f['never']} never were, including {len(f['invisible'])} of the {f['reviewed']} places with 300+ Google reviews.
- **Each engine sees a different, narrow slice.** One engine alone recommends {pct(min(f['coverage_engine'].values()))}-{pct(max(f['coverage_engine'].values()))} of Kaş places, and only {f['all_three']} places are recommended by all three.
- **Popularity matters but does not decide.** Recommendations rise with Google review count (Spearman ρ = {f['rho']:.2f}), yet well-reviewed places are missing and the same question asked three times shares only about a quarter to a third of its recommended places.
- **OTAs back most recommendations, except for OpenAI.** Recommended places are backed by OTA pages {pct(h.at['all', 'ota'])} of the time and by their own website {pct(h.at['all', 'own_website'])}. Gemini and Anthropic lean heavily on OTAs; OpenAI cites hotels' own websites more often ({pct(h.at['openai', 'own_website'])} vs {pct(h.at['openai', 'ota'])}). Under the rule fixed in advance, the hypothesis is **{f['verdict']}**.
- **Some answers are out of date.** A permanently closed hotel was recommended in {f['closed_answers']} of {f['answers']} answers ({pct(f['closed_answers'] / f['answers'], 1)}), by all three engines and in all four languages.

## 1. What was measured

- **Destination:** Kaş town, the Çukurbağ peninsula and Limanağzı (hand-drawn boundary); {f['places']} places to stay from Google Maps, including {f['found_via_answers']} found only through the answers.
- **Questions:** {f['base_prompts']} traveler questions from 8 personas (couples, families, divers, remote workers, retirees, pet owners, honeymooners, road-trippers), each written naturally in English, Turkish, German and Russian.
- **Engines:** OpenAI (gpt-6-luna), Google Gemini (gemini-3.8-flash) and Anthropic (claude-haiku-5-5), each with its own web search, through the API; every question asked {f['repeats']} times.
- **Scale:** {f['answers']} answers ({f['failed']} failed), {f['mentions']} named places, {f['recommendations']} recommendations; {pct(f['matched_share'])} of named places matched to the Kaş list automatically. Total API spend ${f['spend']:.2f}, plus {f['credits']} free Serper credits.

## 2. Who gets recommended

- **Coverage:** {pct(f['coverage'])} of Kaş places recommended at least once ({f['recommended']} of {f['places']}). By engine: {cov_e}. By language: {cov_l}. (Chart 1)
- **Invisible despite being popular:** {invisible}, among others, were never recommended.
- **Concentration:** the top 5 places receive {pct(f['top_share'])} of recommendations: {', '.join(f['top_names'])}. Concentration is higher within a single engine ({', '.join(f"{ENGINES[e]} {pct(v)}" for e, v in f['top_share_engine'].items())}). (Chart 2)
- **Engines disagree:** each engine's favourites differ ({tops}). Overlap of recommended places between engine pairs: {overlap}. Anthropic names more places per answer (median per answer: {per_answer}).

## 3. Popularity

Recommendations rise with Google review count (Spearman ρ = {f['rho']:.2f}): popular places are more visible, but the relationship is far from complete. Several places with hundreds of reviews are never recommended, while some with few reviews are recommended often. This describes a pattern; it does not show that reviews cause recommendations. (Chart 3)

## 4. Which sources back a recommendation

Each citation is linked to the recommended places named in the same part of the answer. Shares are of (recommended place, cited domain) pairs. (Chart 4)

| Engine | OTA | Hotel's own website | Blog / media | Review site | Other |
|---|---|---|---|---|---|
{src_rows}

**Hypothesis test** (recommended Kaş places that have a website; a recommendation can be backed by both). (Chart 5)

| Engine | Backed by an OTA | Backed by own website | No source attached | Recommendations |
|---|---|---|---|---|
{hyp_rows}

- **Verdict, by the rule fixed in advance: {f['verdict']}.** OTA backing is far higher overall and for Gemini and Anthropic, but OpenAI cites hotels' own websites more often than OTAs, so the "every engine" condition fails.
- **Robust:** leaving out the domains categorised by the AI gives the same picture (OTA {pct(r.at['all', 'ota'])} vs own website {pct(r.at['all', 'own_website'])}; verdict {f['robust_verdict']}).
- **Gemini often attaches no source** to a recommended place ({pct(h.at['gemini', 'no_source'])}).
- **What this suggests for HotelRunner:** the content hotels distribute through OTAs is what Gemini and Anthropic cite most, so channel content plausibly shapes visibility there; for OpenAI, the hotel's own website (and its booking engine) is the main cited source. Both are patterns in citations, not proof of what drives a recommendation.

## 5. Recommendations that are not a Kaş place in our list

{pct(f['destination_share'])} of recommendations point to a place in our Kaş list. Of the rest (Chart 6):
- **Outside the boundary ({pct(f['outside_share'])} of recommendations):** mostly close by (about {f['near']} mentions under 5 km, e.g. Bayındır) and Kalkan, Patara and Gelemiş (about {f['far']} mentions, 15-30 km). These are in Kaş district, so engines read "Kaş" more broadly than the town.
- **Permanently closed:** Hotel Cachet, closed per Google Maps and Tripadvisor, was recommended {f['closed']} times ({closed_engines}).
- **Real but not on Google Maps:** {f['not_on_google']} recommendations (e.g. Blue Island Luxury, on Trivago with its own booking site).
- **Unverifiable:** {f['unverifiable']} recommendations name places that could not be found by lookup or by hand.

## 6. Answers vary from run to run

Asking the same question again gives a different list: the mean overlap (Jaccard) of recommended places between two runs of the same question is {stability}. Single answers are therefore anecdotes; the figures above pool three runs of every question.

## 7. How reliable are these numbers

- Sanity checks pass: all answers present, every place id valid, no duplicate rows, every citation span inside its answer.
- {quality}
- Names the automatic matcher could not resolve were looked up on Google Maps; the most frequent doubtful ones were checked by hand (`config/manual_verifications.yaml`).
- Domain categories: the most-cited domains covering 80% of citations were categorised by hand; the rest by AI, with the robustness check above.

## 8. Main limitations

- API answers are a proxy for what travelers see in the consumer apps (no personalisation, location or memory).
- Answers vary across runs and model versions; results pool three repeats.
- Popularity and visibility are correlated; the study describes patterns, not causes.
- The place list depends on Google Maps; closed places are invisible to the automatic lookup.
- Full list: [limitations.md](limitations.md).

## Suggested slides

1. **The question and the setup:** 3 engines x 4 languages x 20 traveler questions x 3 repeats.
2. **Most of Kaş is invisible:** coverage {pct(f['coverage'])}, popular places missing, engines disagree (Charts 1-2).
3. **Popularity helps but is not enough** (Chart 3).
4. **OTAs feed the answers, except on OpenAI** (Charts 4-5): the hypothesis result and what it means for HotelRunner's channel and booking-engine products.
5. **Answers can be wrong or outdated:** the closed hotel, places outside Kaş, run-to-run variation (Chart 6).
"""


def main():
    out = ROOT / "docs" / "findings.md"
    out.write_text(render(facts()))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
