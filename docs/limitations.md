# Limitations (running log)

Kept from day 1 for the presentation and later analysis.

## Measurement
- **API ≠ consumer apps.** Calls have no personalization, location, memory or system prompt. Results are a proxy for what travelers see. A traveler's location (e.g. a German user) is not simulated because Gemini grounding does not support it, so engines would not be comparable.
- **Variability.** Answers vary across runs and model versions; we report 3 repeats and log model id + timestamp per call.
- **Correlation, not causation.** Popularity (review count) strongly influences recommendations; the PoC describes patterns and does not prove causes.
- **Cited ≠ consulted.** Only URLs the answer cites are counted; engines may read more than they cite.

- **Trip dates are a month and year, not exact dates.** Prompts say e.g. "in September 2027" (the year is added by fixed grammar rules, since every month had passed in 2026). In the pilot, 71% of answers hedged on prices without exact dates (OpenAI and Anthropic 100%, Gemini 12%) and 19% closed by offering to refine if given exact dates. This crossed the agreed 10% threshold for adding a topic to the trip context, but exact dates were deliberately not added: every pilot answer still recommended specific places (the rule's purpose), the questions were closing offers rather than blockers, and exact dates would push engines toward availability checks they cannot perform.

## Place list
- Depends on Google Maps' coverage of Kaş.
- Collected with "hotel", "otel", "pansiyon" and "apart" searches on a 46-point grid. Against OpenStreetMap, 77 of 104 named places in Kaş were found (74%). Of the missing places with "hotel" in the name, 3 were confirmed absent from Google Maps (likely closed or renamed).
- Searches stop paging when a page adds no new places. In the dense town center, page 1 repeats the same prominent hotels, so lower-ranked places can be missed (Kekova Hotel was missed by the hotel searches this way and found only through "pansiyon"). Extra pages were not fetched to conserve free credits.
- 21 places missed by the grid search (e.g. Carruba Boutique Hotel, 245 Google reviews) were found through the answers themselves and added to the list. Places that no answer named and the grid missed remain absent, so coverage may still be slightly overstated.
- Google lists some properties once per unit (e.g. "Kale Apart - Studio with Sea View"); 70 such or duplicate listings were merged into their property by name prefix within 300 m. A few unit listings with unusual names may remain.
- Google categories are owner-chosen and inconsistent (many pensions are listed as hotels).
- Destination boundary is a hand-drawn polygon (Kaş town + Çukurbağ peninsula); coverage is sensitive to it, so distance from center is stored for every place.

## Matcher accuracy
- The hand-check covered 30 answers (2-3 per engine x language, fixed seed) with 132 matched places; the project owner found no errors, giving precision and recall of 100% with 95% Wilson intervals of 97.2-100%. The check was done by one person as an overall review of the sample, and mentions within one answer are not independent, so the true error rate may be somewhat higher than the interval suggests. The sample was drawn before 21 found places were added to the list; those changes do not affect matched rows.

## Unmatched places
- **Closed places are invisible to the automatic lookup.** Serper's Google Maps results omit permanently closed places entirely (no result and no status field), although Google itself shows them as "Permanently closed". Automatic categories "osm_only" and "unverifiable" can therefore contain closed places. Example: Hotel Cachet, recommended in 8 of 48 pilot answers, is permanently closed per Google Maps and Tripadvisor (checked by hand on 2026-10-22).
- **18% of mentions do not match the Kaş place list (82% match).** This is acceptable for the PoC: most of the remainder is very likely in Kaş municipality/district but outside our chosen town boundary (e.g. Kalkan, Patara, Gelemiş), and the boundary decision is revisited after the results. To save Serper credits, only the 80 unmatched names mentioned 2+ times were looked up (they hold about 80% of unmatched mentions); 155 names mentioned once are labelled "not looked up".
- Hand-verified facts are kept in `config/manual_verifications.yaml` (with sources and date) and override the lookup. Only the most-mentioned osm_only/unverifiable names are checked by hand, so the remaining ones may still include closures.

## Scope
- Kaş is almost entirely independent properties; results may not carry over to chain-heavy destinations (Bodrum, Antalya city).
