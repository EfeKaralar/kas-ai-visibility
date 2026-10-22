# Metric definitions

Written on 2026-10-22, **before** any metric was computed on the full run and without looking at citation categories or source shares in the pilot. Definitions change only to fix bugs or when a rule proves unworkable on the full data; every change is logged at the end with date and reason.

## Units

- **Answer:** one engine response to one prompt in one language, for one repeat. The main run has 720 (20 base prompts x 4 languages x 3 engines x 3 repeats). Failed answers are excluded and counted.
- **Destination population:** `destination_places()`, i.e. accommodation inside the approved polygon (458 places). Every per-place denominator comes from here; `check_population()` must pass.
- **Mention:** one place named in one answer (one row per place per answer). After matching it has a label (`recommended`, `negative`, `incidental`) and, if matched, a `place_uid`.
- **Recommendation:** a mention labelled `recommended`. Unless stated otherwise, metrics use recommendations only.
- **Matcher corrections:** an unmatched name that the lookup finds in the place list (`in_list_missed_by_matcher`) is treated as matched to that place.
- **Base prompt:** one of the 20 intents; it has 4 language versions.

## Visibility metrics

| # | Metric | Definition |
|---|---|---|
| V1 | Recommendations per place | Number of answers recommending the place. Reported overall and by engine, language and persona. |
| V2 | Distinct base prompts per place | Number of distinct base prompts with at least one answer recommending the place. Shows breadth rather than repetition. |
| V3 | Coverage | Share of destination places recommended in at least one answer. Overall, per engine and per language. |
| V4 | Concentration | Share of all recommendations of destination places that go to the 5 most-recommended places. Overall and per engine. |
| V5 | Popularity | Scatter of V1 against Google review count (log scale), all destination places including those never recommended. Spearman rank correlation reported as a description, not a causal estimate. |
| V6 | No-recommendation rate | Share of answers with no recommendation at all. Per engine and language. |
| V7 | Where recommendations point | Share of all recommendations that are: destination places; listed places outside the destination; and each unmatched category (outside destination, not in place list, permanently closed, osm only, unverifiable, not accommodation). |
| V8 | Repeat stability | For each prompt x engine, the mean Jaccard overlap of recommended place sets between pairs of repeats. Reported per engine. |

## Source metrics

### Domain categories
Each cited domain gets one category: **OTA** (booking and travel agencies, including Turkish ones, and metasearch such as Trivago), **review site**, **own website**, **map**, **blog/media**, **other**.

- **Own website** is decided per place, not per domain: a citation is the place's own website when its domain equals the website domain of that place in the place list (domains shared by several places, such as instagram.com, never count as own website). In S1 (per answer) "own website" means the website of any listed place or any domain hand-categorised as a hotel's website; in S2 and H (per recommendation) it means the recommended place's own site, and another hotel's site counts as "other".
- **Rule for the hand-made list:** after the full run, the most-cited domains that together account for 80% of all citations are categorised by hand in `config/domains.yaml`. The remaining domains are categorised by the helper LLM and flagged `source: llm`.
- **Robustness check:** every source metric is recomputed with LLM-categorised domains left out; if a conclusion changes, the report says so.

### Linking a citation to a recommendation
A citation **supports** a recommended place when its text span overlaps the place's **segment**: the text from the start of the line where the place is first named to the start of the line where the next different place is first named (or the end of the answer). This covers headings followed by a description, bullet lists and table rows. A segment with no overlapping citation is "no source attached". Several places in one segment share its citations.

| # | Metric | Definition |
|---|---|---|
| S1 | Sources per answer | For each category, share of answers that cite it at least once. Per engine and language. |
| S2 | Sources per recommendation (headline) | Over all (recommendation, domain) pairs, the share in each category, counting a domain once per recommendation however often it is cited. Plus the share of recommendations with no source attached. Per engine and language. |

### Hypothesis test (H)
Population: recommendations of destination places that have a website in the place list (so an own-website citation is possible). For each such recommendation: does at least one supporting citation come from (a) the place's own website, (b) an OTA?

Report the share with own website, with OTA, with both, with neither (other sources only), and with no source, overall and per engine.

**Decision rule, fixed in advance:**
- **Supported:** OTA-backed share is higher than own-website-backed share overall **and** in every engine.
- **Not supported:** own-website-backed share is equal to or higher than OTA-backed share overall.
- **Mixed:** anything else (e.g. holds overall but not for one engine). Reported as mixed, not as support.

The test is descriptive; it does not show that OTA content *causes* recommendations.

## Matcher quality (hand-check)
- **Sample:** 30 answers from the main run, stratified by engine x language (12 groups, 2 or 3 answers each), fixed random seed.
- **Labelling:** the project owner marks each extracted mention as correct or not (place and label), and adds places the matcher missed.
- **Precision:** correct matched mentions / all matched mentions in the sample. **Recall:** correct matched mentions / (correct matched mentions + missed places). Both reported with a 95% Wilson interval.

## Sanity checks (run before reporting)
- All 720 answers present or explicitly failed; failures counted per engine.
- Every recommendation's place_uid exists in the place list; destination metrics pass `check_population()`.
- Each answer x place appears at most once in the mentions table.
- Citation spans lie within their answer text.

## Change log
- 2026-10-22: first version.
- 2026-10-22 (after the full run): no definition changed. The citation-span sanity check found 14 zero-length spans in one Anthropic answer (citations on an empty text block); the extractor now gives such citations the span of the preceding text. This is an extraction bug fix, not a metric change. The robustness check runs as specified and gives the same verdict.
- 2026-10-22 (data correction, owner-approved): 21 real Kaş places that the grid search missed but the unmatched-name lookup found on Google Maps (inside the polygon, accommodation) were added to the place list (`source: answer_lookup`), using cached Google data. The destination population grows from 458 to 479; their 110+ mentions now count as matched through the existing matcher-correction rule. No definition changed. Effect: coverage 43.7% -> 46.1%; top-5 share 18.1% -> 17.4%; Spearman 0.61 -> 0.59; hypothesis shares within 1 point; verdict unchanged (mixed).
