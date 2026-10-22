# Kaş AI-visibility PoC — agreed plan

**Question:** when travelers ask AI assistants about Kaş, which places get recommended, which are invisible, and which sources feed those answers?

**Hypothesis:** AI answers draw mostly on OTA and third-party pages rather than places' own websites, so the content HotelRunner distributes shapes AI visibility.

## Scope and data
- **Destination:** Kaş town + Çukurbağ peninsula, hand-drawn polygon (`config/kas_polygon.geojson`). Kalkan, Meis/Kastellorizo, inland villages = outside.
- **Place list:** Google Maps via Serper.dev, keywords "hotel"/"otel"/"pansiyon"/"apart" over a 46-point grid, paging (max 3) until a page adds nothing new. "pansiyon"/"apart" were added after an OSM cross-check showed small properties missing. All returned accommodation types kept and analysed; single-unit listings merged into their property. Extra pages deferred to conserve credits (Serper's paid plans also start at $50).
- **Google Maps data provider:** Serper.dev on its free 2,500 signup credits (no card). DataForSEO was dropped because account activation requires a $50 payment. Serper usage is capped at 2,000 credits, tracked in the budget ledger; the rest is reserved for name lookups.
- **Unmatched recommendations:** Serper Maps lookup per unique name (with and without "Kaş", LLM judge on top results, OSM fallback) → outside / not in place list / matcher miss / not accommodation / osm_only / unverifiable.
- **Manual closure check (standing step before reporting):** Serper omits permanently closed places, so the most-mentioned osm_only and unverifiable names are checked by hand on Google Maps / Tripadvisor and recorded in `config/manual_verifications.yaml` (status, sources, date), which overrides the lookup. First case: Hotel Cachet, permanently closed.
- **Stretch goal data source:** open. Serper does not return hotel amenities; candidates are SerpApi's free 250 searches/month (Google Hotels property details) or Google Places API. Decide when the stretch goal starts.

- **Boundary decision (2026-10-22): keep the town boundary.** Answers also recommend places outside it (mostly within 5 km, e.g. Bayındır, plus Kalkan, Patara and Gelemiş), all in Kaş district. This is reported as a finding (engines read "Kaş" as the district) rather than widening the destination, which would need a new place collection.

## Prompts
- 8 personas, ~10 attributes (list decided together before Step 2), 20 base prompts × TR/EN/DE/RU = 80 prompts. Same intent per language, written naturally. Every prompt names Kaş.
- Trip context (month, nights, party, budget band) and allowed attributes are fixed per persona in config, so attributes and context always match.
- Budget is a `{budget}` placeholder: € for EN/DE/RU, TL at `eur_try_rate` for TR. Change the rate only before the main run.
- All 80 prompts are reviewed before any engine call.

## Engines
- OpenAI mini + web search, Gemini Flash + Google Search grounding, Claude Haiku 5.5 + web search, via LangChain (pinned).
- Plain calls: no system prompt, no location, default temperature. 3 repeats → 720 calls.
- Day-1 parity check: LangChain vs native SDK URL lists per engine; native SDK fallback for any engine that loses URLs.
- Gemini redirect URLs resolved immediately. Disk cache keyed by request hash. Live-box queries cached separately and excluded from metrics.
- Pilot of ~50 calls: clarifying-question topics labelled; any topic >10% gets added to trip context (max one re-pilot). Automatic follow-ups only if >20% of answers recommend nothing, stored separately.

## Matching (DeepSeek V3.2, non-thinking)
- One structured-output call per answer, with the full place list: named places, position, matched place id or none, label `recommended` / `negative` / `incidental`. Metrics use `recommended` only.
- "Recommended" = offered as an option for the traveler's request (lists included); not warnings, landmarks or comparisons.
- Fuzzy matching (with transliteration) flags disagreements. 30 answers hand-labelled, stratified by engine × language; precision and recall reported with margins.

## Metrics
- Recommendations and distinct prompts per place, by language / engine / persona. Coverage, top-5 concentration, recommendations vs review count, no-recommendation rate.
- Citations, two views: per recommendation (headline; citation linked to the place by sentence/paragraph position, plus "no source attached") and per answer (domain cited at all). Cited URLs only.
- Domain categories: top domains by hand, long tail by DeepSeek (flagged); conclusions checked without LLM labels.
- Hypothesis metric: when a place is recommended, how often its own website vs an OTA backs it.

## Budget
One running total for all API spend; warn at $30, hard stop at $50. Estimate ~$15–20.

## Outputs (English)
Streamlit app (6 pages + live-query box), 6 PNG charts, findings summary, `docs/limitations.md`.

Charts: coverage by engine; top-5 concentration + long tail; recommendations vs review count; citation categories by engine/language; own site vs OTA per recommendation; unmatched by category.

## User gates
(a) approve polygon · (b) personas and attributes · (c) review 80 prompts · (d) label 30 answers.

## Later
Stretch (listing attribute check) only after Steps 1–5.
