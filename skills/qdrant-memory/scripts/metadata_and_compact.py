#!/usr/bin/env python3
"""Metadata + Compaction pipeline.

This script is designed to be run on a schedule (cron). It will:
1) Detect if anything new exists in Redis buffer since last run.
2) If new content exists, generate:
   - title
   - tags
   - entities
   - category
   - compact summary
   using a cheap LLM (Minimax m2.5) with fallback (Gemini Flash)
3) Store the metadata + summary into Qdrant as a single point (collection: kimi_kb by default)
   while leaving raw transcripts in files/Redis.

It is intentionally conservative: if nothing new, it exits quickly.

Env:
  REDIS_HOST/REDIS_PORT
  QDRANT_URL
  QDRANT_META_COLLECTION (default: kimi_kb)
  OPENROUTER_API_KEY (required for LLM)
  LLM_PRIMARY_MODEL / LLM_FALLBACK_MODEL

Usage:
  python3 metadata_and_compact.py --user-id michael
  python3 metadata_and_compact.py --user-id michael --max-items 200

"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime

import redis

from llm_router import chat_json

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
META_COLLECTION = os.getenv("QDRANT_META_COLLECTION", "kimi_kb")

STATE_DIR = os.getenv("MEMORY_STATE_DIR", os.path.join(os.path.expanduser("~"), ".openclaw", "memory_state"))

SYSTEM_PROMPT = (
    "You are a metadata extractor and compactor for conversation logs. "
    "Return STRICT JSON with keys: title (string), category (string), "
    "tags (array of short lowercase hyphenated strings), entities (array of strings), "
    "summary (string, <= 1200 chars). "
    "Prefer 6-14 tags. Tags should be searchable facets (client/project/infra/topic)."
)


def _state_path(user_id: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"meta_state_{user_id}.json")


def load_state(user_id: str) -> dict:
    p = _state_path(user_id)
    if not os.path.exists(p):
        return {"last_redis_len": 0, "updated_at": None}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_redis_len": 0, "updated_at": None}


def save_state(user_id: str, st: dict) -> None:
    p = _state_path(user_id)
    st["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(p, "w") as f:
        json.dump(st, f, indent=2, sort_keys=True)


def redis_get_new_items(user_id: str, max_items: int, last_len: int):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    key = f"mem:{user_id}"
    cur_len = r.llen(key)
    if cur_len <= last_len:
        return [], cur_len

    # Only grab the delta (best effort). Our list is chronological if RPUSH is used.
    start = last_len
    end = min(cur_len - 1, last_len + max_items - 1)
    items = r.lrange(key, start, end)
    turns = []
    for it in items:
        try:
            turns.append(json.loads(it))
        except Exception:
            continue
    return turns, cur_len


def qdrant_upsert(point_id: str, vector, payload: dict):
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
    import urllib.request

    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{META_COLLECTION}/points?wait=true",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        out = json.loads(resp.read().decode("utf-8"))
        return out.get("status") == "ok"


def ollama_embed(text: str):
    # Uses the same Ollama embed endpoint as auto_store
    import urllib.request

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1")
    data = json.dumps({"model": "snowflake-arctic-embed2", "input": text[:8192]}).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url}/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode("utf-8"))
        return out["data"][0]["embedding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--max-items", type=int, default=200)
    args = ap.parse_args()

    st = load_state(args.user_id)
    last_len = int(st.get("last_redis_len", 0))

    turns, cur_len = redis_get_new_items(args.user_id, args.max_items, last_len)
    if not turns:
        print("No new turns; skipping")
        return

    # Build compact source text
    lines = []
    for t in turns:
        role = t.get("role", "")
        content = t.get("content", "")
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    source_text = "\n".join(lines)

    meta = chat_json(SYSTEM_PROMPT, source_text[:24000])

    # basic validation
    for k in ("title", "category", "tags", "entities", "summary"):
        if k not in meta:
            raise SystemExit(f"Missing key in meta: {k}")

    summary = str(meta.get("summary", ""))[:2000]
    emb = ollama_embed(summary)

    payload = {
        "user_id": args.user_id,
        "title": str(meta.get("title", ""))[:200],
        "category": str(meta.get("category", ""))[:120],
        "tags": meta.get("tags", [])[:30],
        "entities": meta.get("entities", [])[:30],
        "summary": summary,
        "source": "redis_delta",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "redis_range": {"from": last_len, "to": cur_len - 1},
    }

    ok = qdrant_upsert(str(uuid.uuid4()), emb, payload)
    if not ok:
        raise SystemExit("Failed to upsert metadata point")

    st["last_redis_len"] = cur_len
    save_state(args.user_id, st)

    print(f"Stored metadata point for {args.user_id} (redis {last_len}->{cur_len})")


if __name__ == "__main__":
    main()
