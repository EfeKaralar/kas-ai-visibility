"""Query the AI engines with the traveler prompts and store answers and citations.

Each (prompt, engine, repeat) is one plain call through LangChain with the engine's own
web-search tool. Responses are cached by request hash and never re-sent; spend goes to the
shared budget ledger and the run stops before the hard cap.

    uv run python -m src.query_engines --pilot     # repeat 1 of four base prompts
    uv run python -m src.query_engines             # every prompt x engine x repeat

Outputs: data/processed/answers.parquet and data/processed/citations.parquet.
Citation positions are character offsets into the answer text, so later steps can link
each citation to the places named near it.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import requests
from langchain.chat_models import init_chat_model

from src.common import (PRICING, PROCESSED, RAW, SETTINGS, Budget, BudgetExceeded, cache_get, cache_put,
                        now_iso, request_hash)

STAGE = "engine_queries"
CACHE = RAW / "engines"
COST_CEILING_PER_CALL = 0.08
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign"}
ledger_lock = threading.Lock()


# --- citation extraction (one function per engine) -------------------------

def clean_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k not in TRACKING_PARAMS])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def text_blocks(message: dict) -> list[dict]:
    content = message["content"]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [block for block in content if block.get("type") == "text"]


def extract_openai(message: dict) -> dict:
    text, citations, offset = "", [], 0
    for block in text_blocks(message):
        for note in block.get("annotations") or []:
            if note.get("type") == "url_citation":
                citations.append({"url": clean_url(note["url"]), "title": note.get("title"),
                                  "start": offset + note["start_index"], "end": offset + note["end_index"],
                                  "cited_text": None})
        text += block["text"]
        offset = len(text)
    calls = [b for b in message["content"] if isinstance(b, dict) and b.get("type") == "web_search_call"]
    consulted = [s["url"] for call in calls for s in (call.get("action") or {}).get("sources") or []]
    return {"text": text, "citations": citations, "searches": len(calls), "consulted": consulted}


def extract_anthropic(message: dict) -> dict:
    text, citations = "", []
    for block in text_blocks(message):
        start = len(text)
        text += block["text"]
        for cite in block.get("citations") or []:
            citations.append({"url": clean_url(cite["url"]), "title": cite.get("title"),
                              "start": start, "end": len(text), "cited_text": cite.get("cited_text")})
    results = [b for b in message["content"] if isinstance(b, dict) and b.get("type") == "web_search_tool_result"]
    consulted = [r["url"] for b in results if isinstance(b.get("content"), list) for r in b["content"]]
    usage = message["response_metadata"].get("usage") or {}
    searches = (usage.get("server_tool_use") or {}).get("web_search_requests", 0)
    return {"text": text, "citations": citations, "searches": searches, "consulted": consulted}


def resolve_redirect(uri: str) -> str | None:
    """Gemini cites vertexaisearch redirect links that expire; resolve them immediately."""
    for _ in range(3):
        try:
            response = requests.head(uri, allow_redirects=False, timeout=20)
            return response.headers.get("Location")
        except requests.RequestException:
            time.sleep(2)
    return None


def extract_gemini(message: dict) -> dict:
    text = "".join(block["text"] for block in text_blocks(message))
    encoded = text.encode("utf-8")
    grounding = message["response_metadata"].get("grounding_metadata") or {}
    chunks = []
    for chunk in grounding.get("grounding_chunks") or []:
        web = chunk.get("web") or {}
        resolved = resolve_redirect(web["uri"]) if web.get("uri") else None
        # Gemini's title is the source domain; keep it so an unresolved link still has a domain
        chunks.append({"url": clean_url(resolved) if resolved else None, "title": web.get("title"),
                       "redirect": web.get("uri")})
    citations = []
    for support in grounding.get("grounding_supports") or []:
        segment = support["segment"]
        # segment offsets are UTF-8 byte offsets; convert to character offsets
        start = len(encoded[:segment.get("start_index") or 0].decode("utf-8", errors="ignore"))
        end = len(encoded[:segment["end_index"]].decode("utf-8", errors="ignore"))
        for index in support.get("grounding_chunk_indices") or []:
            chunk = chunks[index]
            citations.append({"url": chunk["url"], "title": chunk["title"], "start": start, "end": end,
                              "cited_text": segment.get("text")})
    return {"text": text, "citations": citations, "searches": len(grounding.get("web_search_queries") or []),
            "consulted": [chunk["url"] for chunk in chunks]}


EXTRACTORS = {"openai": extract_openai, "anthropic": extract_anthropic, "gemini": extract_gemini}


# --- cost ------------------------------------------------------------------

def gemini_free_searches_left(budget: Budget, model: str) -> int:
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    used = 0
    if budget.ledger.exists():
        for line in budget.ledger.read_text().splitlines():
            row = json.loads(line)
            if row.get("model") == model and row["timestamp"].startswith(month):
                used += row.get("searches", 0)
    return max(0, PRICING["llm"][model].get("free_searches_per_month", 0) - used)


def answer_cost(model: str, usage: dict, searches: int, budget: Budget) -> float:
    price = PRICING["llm"][model]
    billable = searches
    if "free_searches_per_month" in price:
        billable = max(0, searches - gemini_free_searches_left(budget, model))
    return ((usage.get("input_tokens", 0) * price["input_per_million"]
             + usage.get("output_tokens", 0) * price["output_per_million"]) / 1_000_000
            + billable * price["per_search"])


# --- querying ----------------------------------------------------------------

def request_for(engine: str, prompt: pd.Series, repeat: int) -> dict:
    config = SETTINGS["engines"][engine]
    return {"engine": engine, "model": config["model"], "init": config["init"], "tools": config["tools"],
            "prompt_id": prompt.prompt_id, "text": prompt.text, "repeat": repeat}


def query(request: dict, budget: Budget, live: bool = False) -> dict:
    """One engine call, cached. Returns the cache record (raw message plus extraction)."""
    folder = CACHE / ("live" if live else request["engine"])
    key = request_hash(request)
    if cached := cache_get(folder, key):
        return cached
    with ledger_lock:
        budget.check(COST_CEILING_PER_CALL)
    config = SETTINGS["engine_queries"]
    model = init_chat_model(request["model"], timeout=config["timeout_seconds"], max_retries=0, **request["init"])
    model = model.bind_tools(request["tools"])

    error, message, started = None, None, time.time()
    for attempt in range(config["retries"] + 1):
        try:
            message = model.invoke(request["text"]).model_dump(mode="json")
            break
        except Exception as exc:  # provider errors vary; record and retry with backoff
            error = repr(exc)[:500]
            time.sleep(2 ** (attempt + 2))
    if message is None:
        return {"request": request, "key": key, "error": error, "fetched_at": now_iso()}

    extracted = EXTRACTORS[request["engine"]](message)
    usage = message.get("usage_metadata") or {}
    with ledger_lock:
        cost = answer_cost(request["model"], usage, extracted["searches"], budget)
        budget.record(STAGE, request["engine"], cost, f"{request['prompt_id']} r{request['repeat']}",
                      model=request["model"], searches=extracted["searches"])
    record = {"request": request, "key": key, "fetched_at": now_iso(), "seconds": round(time.time() - started, 1),
              "model_id": message["response_metadata"].get("model_name"),
              "response_id": message.get("id"), "usage": usage, "cost_usd": cost,
              "extracted": extracted, "message": message}
    cache_put(folder, key, record)
    return record


def to_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    answers, citations = [], []
    for record in records:
        request = record["request"]
        answer_id = f"{request['prompt_id']}_{request['engine']}_r{request['repeat']}"
        extracted = record.get("extracted") or {}
        answers.append({"answer_id": answer_id, "prompt_id": request["prompt_id"], "engine": request["engine"],
                        "repeat": request["repeat"], "model": request["model"], "model_id": record.get("model_id"),
                        "fetched_at": record["fetched_at"], "seconds": record.get("seconds"),
                        "error": record.get("error"), "text": extracted.get("text"),
                        "searches": extracted.get("searches"), "n_citations": len(extracted.get("citations", [])),
                        "n_consulted": len(extracted.get("consulted", [])), "cost_usd": record.get("cost_usd")})
        for order, cite in enumerate(extracted.get("citations", [])):
            url = cite["url"]
            domain = urlsplit(url).netloc.removeprefix("www.") if url else (cite["title"] or "").lower()
            citations.append({"answer_id": answer_id, "order": order, "url": url, "domain": domain,
                              "url_resolved": url is not None, **{k: cite[k] for k in ("title", "start", "end",
                                                                                         "cited_text")}})
    return pd.DataFrame(answers), pd.DataFrame(citations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="repeat 1 of the pilot base prompts only")
    args = parser.parse_args()
    config, budget = SETTINGS["engine_queries"], Budget()
    prompts = pd.read_parquet(PROCESSED / "prompts.parquet")
    repeats = range(1, config["repeats"] + 1)
    if args.pilot:
        prompts = prompts[prompts.base_id.isin(config["pilot_base_prompts"])]
        repeats = range(1, 2)
    requests_ = [request_for(engine, prompt, repeat) for prompt in prompts.itertuples()
                 for engine in SETTINGS["engines"] for repeat in repeats]
    print(f"{len(requests_)} calls ({len(prompts)} prompts x {len(SETTINGS['engines'])} engines x "
          f"{len(repeats)} repeats); spent so far ${budget.spent():.2f}")

    records = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(query, request, budget) for request in requests_]
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if done % 10 == 0 or done == len(futures):
                    print(f"  {done}/{len(futures)} done, spent ${budget.spent():.2f}")
        except BudgetExceeded as stop:
            print(f"STOPPED: {stop}")
            pool.shutdown(cancel_futures=True)

    answers, citations = to_tables(records)
    suffix = "_pilot" if args.pilot else ""
    answers.to_parquet(PROCESSED / f"answers{suffix}.parquet", index=False)
    citations.to_parquet(PROCESSED / f"citations{suffix}.parquet", index=False)
    failed = answers[answers.error.notna()]
    print(f"\nAnswers: {len(answers)} ({len(failed)} failed); citations: {len(citations)} "
          f"({(~citations.url_resolved).sum() if len(citations) else 0} unresolved)")
    print(answers.groupby("engine")[["seconds", "searches", "n_citations", "cost_usd"]].mean().round(3).to_string())
    print(f"Spend so far: ${budget.spent():.2f}")


if __name__ == "__main__":
    main()
