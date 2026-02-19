#!/usr/bin/env python3
"""LLM Router for cheap metadata + compaction.

Goal:
- Prefer Minimax m2.5 for tagging + compaction.
- Fallback to Gemini Flash (or any other OpenRouter model) if Minimax fails.

This uses OpenRouter's OpenAI-compatible API.

Env:
  OPENROUTER_API_KEY   (required)
  OPENROUTER_BASE_URL  default: https://openrouter.ai/api/v1
  LLM_PRIMARY_MODEL    default: openrouter/minimax/minimax-m2.5
  LLM_FALLBACK_MODEL   default: openrouter/google/gemini-2.5-flash
  LLM_TIMEOUT          default: 60

Notes:
- We keep this dependency-light (urllib only).
- We request strict JSON when asked.
"""

import json
import os
import sys
import urllib.request

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "openrouter/minimax/minimax-m2.5")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openrouter/google/gemini-2.5-flash")
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


def _post_chat(model: str, messages, response_format=None, temperature=0.2):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        body["response_format"] = response_format

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def chat_json(system: str, user: str) -> dict:
    """Return parsed JSON object. Try primary then fallback."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            resp = _post_chat(model, messages, response_format={"type": "json_object"}, temperature=0.2)
            content = resp["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"LLM failed on both primary and fallback: {last_err}")


def chat_text(system: str, user: str) -> str:
    """Return text. Try primary then fallback."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            resp = _post_chat(model, messages, response_format=None, temperature=0.2)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"LLM failed on both primary and fallback: {last_err}")


if __name__ == "__main__":
    # tiny self-test
    if len(sys.argv) > 1 and sys.argv[1] == "--ping":
        out = chat_json("Return JSON with key ok=true", "ping")
        print(json.dumps(out))
