# Kaş AI-visibility PoC

When travelers ask AI assistants where to stay in Kaş, which places get recommended, which are invisible, and which sources feed those answers? See [docs/plan.md](docs/plan.md) for the agreed design and [docs/limitations.md](docs/limitations.md) for caveats.

## Setup

```bash
uv sync
cp .env.example .env   # fill in API keys
```

Findings: [docs/findings.md](docs/findings.md) · charts: [docs/charts/](docs/charts/)

## Pipeline

| Step | What it does | Command | Output |
|---|---|---|---|
| 0 | Render the destination polygon with OSM accommodation for review | `uv run python -m src.review_polygon` | `data/review/polygon_map.html` |
| 1 | Collect accommodation from Google Maps (Serper.dev, within its free credits), deduplicate, flag in-destination | `uv run python -m src.collect_places` | `data/processed/places.parquet`, `place_search_saturation.csv` |
| 2 | Generate traveler prompts (20 base x en/tr/de/ru) from config/personas_attributes.yaml | `uv run python -m src.generate_prompts` | `data/processed/prompts.parquet`, `data/review/prompts_review.md` |
| 3a | Pilot: repeat 1 of four base prompts on all engines | `uv run python -m src.query_engines --pilot` | `answers_pilot.parquet`, `citations_pilot.parquet` |
| 3b | Label pilot answers (recommendations, clarifying questions, date hedging) | `uv run python -m src.review_pilot` | `pilot_labels.parquet` |
| 3c | Query all prompts x engines x 3 repeats | `caffeinate -i uv run python -m src.query_engines` | `answers.parquet`, `citations.parquet` |
| 4a | Match named places to the list (LLM + hand-verified aliases + own-website citations) | `uv run python -m src.match_mentions` | `mentions.parquet` |
| 4b | Classify unmatched names (Google Maps lookup, LLM judge, OSM, manual verifications) | `uv run python -m src.classify_unmatched --min-mentions 2` | `unmatched_places.parquet` |
| 4c | Add real Kaş places found only through answers to the place list, then re-run 4b | `uv run python -m src.add_found_places` | `places.parquet` (source `answer_lookup`) |
| 4d | Hand-check sample and scoring | `uv run python -m src.hand_check prepare` / `score` | `data/review/hand_check.*`, `metrics/matcher_quality.parquet` |
| 5a | Categorise cited domains (hand list for the 80% head, LLM for the tail) | `uv run python -m src.categorize_domains` | `domains.parquet` |
| 5b | Compute metrics (definitions in docs/metrics.md) | `uv run python -m src.compute_metrics` | `data/processed/metrics/` |
| 5c | Export slide charts | `uv run python -m src.make_charts` | `docs/charts/` |
| 5d | Write the findings summary (every number computed from the metric tables) | `uv run python -m src.write_findings` | `docs/findings.md` |
| 6 | Streamlit app: overview, hotels, outside the list, sources, live query, method | `uv run streamlit run app/app.py` | browser |

Hand-verified facts about places live in `config/manual_verifications.yaml`; hand categories for domains in `config/domains.yaml`. Every step except 1, 3 and the first run of 4b reads only cached data.

### Stopping and resuming the engine run
Every answer is saved the moment it arrives, and a re-run only makes the calls not yet saved, so stopping never loses work or pays twice. Ctrl+C (or `kill`) stops new calls and lets the few in-flight calls, already paid for, finish and be saved; press Ctrl+C again to abort those too. Sleep or a lost connection makes in-flight calls fail; they are retried on the next run. Run the same command again to continue.

## Conventions
- All API spend goes through one budget ledger (`data/budget_ledger.jsonl`): warning at $30, hard stop at $50.
- Every external request is cached on disk under `data/raw/`, keyed by a hash of the request; cached requests are never re-sent.
- `data/` and `.env` are not committed.
