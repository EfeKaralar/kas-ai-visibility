"""Serper.dev client for Google Maps searches, kept inside the free signup credits.

Serper bills in credits, not dollars. Spend is tracked in the shared budget ledger
(at $0, since the free credits need no payment) and capped separately in credits.
"""

import json

import requests

from src.common import RAW, SETTINGS, Budget, BudgetExceeded, cache_get, cache_put, now_iso, request_hash, require_env

MAPS_ENDPOINT = "https://google.serper.dev/maps"
CACHE = RAW / "serper" / "maps"


class Serper:
    def __init__(self, stage: str, budget: Budget):
        (self.api_key,) = require_env("SERPER_API_KEY")
        self.stage, self.budget = stage, budget
        self.config = SETTINGS["serper"]

    def credits_used(self) -> int:
        if not self.budget.ledger.exists():
            return 0
        rows = [json.loads(line) for line in self.budget.ledger.read_text().splitlines() if line]
        return sum(row.get("credits", 0) for row in rows if row["provider"] == "serper")

    def maps_search(self, query: str, lat: float, lon: float, page: int = 1, zoom: str | None = None) -> dict:
        """One Google Maps results page around a coordinate; cached by request."""
        request = {"q": query, "ll": f"@{lat},{lon},{zoom or self.config['zoom']}",
                   "hl": self.config["language"], "page": page}
        key = request_hash(request)
        if cached := cache_get(CACHE, key):
            return cached

        cap, expected = self.config["credit_cap"], self.config["credits_per_maps_call"]
        if self.credits_used() + expected > cap:
            raise BudgetExceeded(f"Serper credit cap {cap} would be exceeded (used {self.credits_used()}).")
        self.budget.check(0)

        response = requests.post(MAPS_ENDPOINT, json=request, timeout=60,
                                 headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"})
        response.raise_for_status()
        body = response.json()
        credits = body.get("credits", expected)
        self.budget.record(self.stage, "serper", 0.0, f"{query} @ {lat},{lon} p{page}", credits=credits)
        record = {"request": request, "fetched_at": now_iso(), "response": body}
        cache_put(CACHE, key, record)
        return record
