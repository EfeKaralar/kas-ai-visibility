# Kaş AI-visibility PoC

When travelers ask AI assistants where to stay in Kaş, which places get recommended, which are invisible, and which sources feed those answers? See [docs/plan.md](docs/plan.md) for the agreed design and [docs/limitations.md](docs/limitations.md) for caveats.

## Setup

```bash
uv sync
cp .env.example .env   # fill in API keys
```

## Pipeline

| Step | What it does | Command | Output |
|---|---|---|---|
| 0 | Render the destination polygon with OSM accommodation for review | `uv run python -m src.review_polygon` | `data/review/polygon_map.html` |
| 1 | Collect accommodation from Google Maps (Serper.dev, within its free credits), deduplicate, flag in-destination | `uv run python -m src.collect_places` | `data/processed/places.parquet`, `place_search_saturation.csv` |

Later steps (prompts, engine queries, matching, metrics, app) are added as they are built.

## Conventions
- All API spend goes through one budget ledger (`data/budget_ledger.jsonl`): warning at $30, hard stop at $50.
- Every external request is cached on disk under `data/raw/`, keyed by a hash of the request; cached requests are never re-sent.
- `data/` and `.env` are not committed.
