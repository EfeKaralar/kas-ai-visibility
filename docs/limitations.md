# Limitations (running log)

Kept from day 1 for the presentation and later analysis.

## Measurement
- **API ≠ consumer apps.** Calls have no personalization, location, memory or system prompt. Results are a proxy for what travelers see. A traveler's location (e.g. a German user) is not simulated because Gemini grounding does not support it, so engines would not be comparable.
- **Variability.** Answers vary across runs and model versions; we report 3 repeats and log model id + timestamp per call.
- **Correlation, not causation.** Popularity (review count) strongly influences recommendations; the PoC describes patterns and does not prove causes.
- **Cited ≠ consulted.** Only URLs the answer cites are counted; engines may read more than they cite.

## Place list
- Depends on Google Maps' coverage of Kaş.
- Collected with "hotel", "otel", "pansiyon" and "apart" searches on a 46-point grid. Against OpenStreetMap, 77 of 104 named places in Kaş were found (74%). Of the missing places with "hotel" in the name, 3 were confirmed absent from Google Maps (likely closed or renamed).
- Searches stop paging when a page adds no new places. In the dense town center, page 1 repeats the same prominent hotels, so lower-ranked places can be missed (Kekova Hotel was missed by the hotel searches this way and found only through "pansiyon"). Extra pages were not fetched to conserve free credits.
- Google lists some properties once per unit (e.g. "Kale Apart - Studio with Sea View"); 70 such or duplicate listings were merged into their property by name prefix within 300 m. A few unit listings with unusual names may remain.
- Google categories are owner-chosen and inconsistent (many pensions are listed as hotels).
- Destination boundary is a hand-drawn polygon (Kaş town + Çukurbağ peninsula); coverage is sensitive to it, so distance from center is stored for every place.

## Scope
- Kaş is almost entirely independent properties; results may not carry over to chain-heavy destinations (Bodrum, Antalya city).
