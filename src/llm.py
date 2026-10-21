"""Provider-agnostic helper-LLM calls via LangChain, with disk cache and budget tracking.

Models are referred to by LangChain's "provider:model" string from config/settings.yaml,
so swapping the helper model is a config change. LangChain does the provider work; this
module only adds what LangChain does not: raw-response caching keyed by request hash and
dollar cost recorded in the shared budget ledger.
"""

import json

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from src.common import PRICING, RAW, SETTINGS, Budget, cache_get, cache_put, now_iso, request_hash

CACHE = RAW / "llm"
COST_CEILING_PER_CALL = 0.02     # pre-call budget check; actual cost is recorded afterwards


def model_name(role: str) -> str:
    return SETTINGS["models"][role]


def call_cost(model: str, usage: dict) -> float:
    price = PRICING["llm"][model]
    return (usage.get("input_tokens", 0) * price["input_per_million"]
            + usage.get("output_tokens", 0) * price["output_per_million"]) / 1_000_000


def structured(role: str, schema: type[BaseModel], messages: list[tuple[str, str]],
               stage: str, budget: Budget) -> BaseModel:
    """One structured-output call; cached requests are never re-sent."""
    model = model_name(role)
    request = {"model": model, "schema": schema.model_json_schema(), "messages": messages}
    key = request_hash(request)
    folder = CACHE / stage
    if cached := cache_get(folder, key):
        return schema.model_validate(cached["parsed"])

    budget.check(COST_CEILING_PER_CALL)
    runnable = init_chat_model(model).with_structured_output(schema, include_raw=True)
    result = runnable.invoke(messages)
    if result["parsing_error"] or result["parsed"] is None:
        raise RuntimeError(f"{model} returned unparseable output: {result['parsing_error']}")
    raw = result["raw"]
    usage = dict(raw.usage_metadata or {})
    cost = call_cost(model, usage)
    budget.record(stage, model.split(":")[0], cost, f"{model} {usage.get('total_tokens', 0)} tokens")
    cache_put(folder, key, {
        "request": request, "fetched_at": now_iso(), "model": model,
        "response_metadata": json.loads(json.dumps(raw.response_metadata, default=str)),
        "usage": usage, "cost_usd": cost, "parsed": result["parsed"].model_dump(),
    })
    return result["parsed"]
