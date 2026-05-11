"""verify_openrouter.py — confirm OPENROUTER_API_KEY works against a free model.

Standalone:
    python scripts/verify_openrouter.py

Programmatic (used by verify_all.py):
    from scripts.verify_openrouter import check
    ok, detail = check()
"""
from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

MODELS_TO_TRY = [
    "meta-llama/llama-3.2-3b-instruct:free",  # primary: free
    "google/gemini-flash-1.5",                # fallback: cheap paid (~$0.0001)
    "openai/gpt-4o-mini",                     # last resort: reliable paid
]


def check() -> tuple[bool, str]:
    """Returns (success, message)."""
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key.startswith("sk-or-v1-..."):
        return False, "OPENROUTER_API_KEY missing or placeholder in .env"
    last_error = None
    for model in MODELS_TO_TRY:
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 10,
                },
                timeout=20.0,
            )
        except httpx.HTTPError as e:
            last_error = f"{model}: HTTPError: {str(e)[:120]}"
            continue
        if r.status_code != 200:
            last_error = f"{model}: status={r.status_code}: {r.text[:120]}"
            continue
        try:
            content = r.json()["choices"][0]["message"]["content"][:60]
        except (KeyError, IndexError, ValueError) as e:
            last_error = f"{model}: bad response shape: {e}"
            continue
        return True, f"model={model}, response={content!r}"
    return False, f"all models failed — last error: {last_error}"


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] OpenRouter — {detail}")
    sys.exit(0 if ok else 1)
