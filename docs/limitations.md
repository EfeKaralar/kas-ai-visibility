# Limitations (running log)

Kept from day 1 for the presentation and later analysis.

## Measurement
- **API ≠ consumer apps.** Calls have no personalization, location, memory or system prompt. Results are a proxy for what travelers see. A traveler's location (e.g. a German user) is not simulated because Gemini grounding does not support it, so engines would not be comparable.
- **Variability.** Answers vary across runs and model versions; we report 3 repeats and log model id + timestamp per call.
- **Correlation, not causation.** Popularity (review count) strongly influences recommendations; the PoC describes patterns and does not prove causes.
- **Cited ≠ consulted.** Only URLs the answer cites are counted; engines may read more than they cite.

## Place list
- Depends on Google Maps' coverage of Kaş.
- Collected with "hotel"/"otel" searches only. Pensions, apart-hotels and villas that Google does not return for these keywords are missing, so their coverage is understated and AI mentions of them land in "not captured in Step 1".
- Google categories are owner-chosen and inconsistent (many pensions are listed as hotels).
- Destination boundary is a hand-drawn polygon (Kaş town + Çukurbağ peninsula); coverage is sensitive to it, so distance from center is stored for every place.

## Scope
- Kaş is almost entirely independent properties; results may not carry over to chain-heavy destinations (Bodrum, Antalya city).
