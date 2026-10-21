"""DataForSEO client: queued (priority) Google Maps searches within the trial credit.

Tasks are posted in batches, then fetched once ready. Posted-but-unfetched task ids
are persisted so an interrupted run resumes without paying for the same search twice.
"""

import json
import time

import requests

from src.common import PRICING, RAW, SETTINGS, Budget, BudgetExceeded, cache_get, cache_put, now_iso, request_hash, require_env

API = "https://api.dataforseo.com/v3"
CACHE = RAW / "dataforseo" / "maps"
PENDING_FILE = RAW / "dataforseo" / "pending_tasks.json"
READY, IN_QUEUE_CODES = 20000, {40601, 40602}
POLL_SECONDS = 15


class DataForSEO:
    def __init__(self, stage: str, budget: Budget):
        self.auth = tuple(require_env("DATAFORSEO_LOGIN", "DATAFORSEO_PASSWORD"))
        self.stage, self.budget = stage, budget
        self.config = SETTINGS["dataforseo"]
        self.unit_cost = PRICING["dataforseo"]["maps_priority_per_call"]

    def _get(self, path: str) -> dict:
        response = requests.get(f"{API}/{path}", auth=self.auth, timeout=120)
        response.raise_for_status()
        return response.json()

    def account_balance(self) -> float:
        return self._get("appendix/user_data")["tasks"][0]["result"][0]["money"]["balance"]

    def provider_spent(self) -> float:
        if not self.budget.ledger.exists():
            return 0.0
        rows = [json.loads(line) for line in self.budget.ledger.read_text().splitlines() if line]
        return sum(row["usd"] for row in rows if row["provider"] == "dataforseo")

    def _guard(self, task_count: int) -> None:
        cost = task_count * self.unit_cost
        self.budget.check(cost)
        cap = self.config["credit_cap_usd"]
        if self.provider_spent() + cost > cap:
            raise BudgetExceeded(f"DataForSEO cap ${cap} would be exceeded "
                                 f"(spent ${self.provider_spent():.3f}, batch ${cost:.3f}).")
        balance = self.account_balance()
        if balance < cost:
            raise BudgetExceeded(f"DataForSEO balance ${balance:.3f} is below batch cost ${cost:.3f}; no top-up planned.")

    def maps_searches(self, tasks: list[dict]) -> list[dict]:
        """Run Google Maps searches; returns one cached record per task, in input order."""
        keys = [request_hash(task) for task in tasks]
        pending = json.loads(PENDING_FILE.read_text()) if PENDING_FILE.exists() else {}
        to_post = [(key, task) for key, task in zip(keys, tasks)
                   if cache_get(CACHE, key) is None and key not in pending]

        batch_size = self.config["tasks_per_post"]
        for start in range(0, len(to_post), batch_size):
            batch = to_post[start:start + batch_size]
            self._guard(len(batch))
            payload = [{**task, "priority": 2, "tag": key} for key, task in batch]
            response = requests.post(f"{API}/serp/google/maps/task_post", json=payload, auth=self.auth, timeout=120)
            response.raise_for_status()
            for (key, task), posted in zip(batch, response.json()["tasks"]):
                if posted["status_code"] != 20100:
                    raise RuntimeError(f"Task post failed: {posted['status_code']} {posted['status_message']}")
                pending[key] = {"id": posted["id"], "task": task}
                self.budget.record(self.stage, "dataforseo", posted.get("cost") or self.unit_cost,
                                   f"{task['keyword']} @ {task['location_coordinate']}")
            PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
            PENDING_FILE.write_text(json.dumps(pending, indent=1))
            print(f"  posted {start + len(batch)}/{len(to_post)} tasks")

        self._collect(pending)
        return [cache_get(CACHE, key) for key in keys]

    def _collect(self, pending: dict) -> None:
        while pending:
            for key, entry in list(pending.items()):
                result = self._get(f"serp/google/maps/task_get/advanced/{entry['id']}")["tasks"][0]
                if result["status_code"] in IN_QUEUE_CODES:
                    continue
                if result["status_code"] != READY:
                    raise RuntimeError(f"Task {entry['id']} failed: {result['status_code']} {result['status_message']}")
                cache_put(CACHE, key, {"request": entry["task"], "fetched_at": now_iso(), "response": result})
                del pending[key]
            PENDING_FILE.write_text(json.dumps(pending, indent=1))
            if pending:
                print(f"  waiting for {len(pending)} tasks...")
                time.sleep(POLL_SECONDS)
