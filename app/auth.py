"""Hardcoded API-key auth (demo): tier + tokens_per_min + fallback model chain."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

API_KEYS: dict[str, dict] = {
    "demo-free": {
        "tier": "free",
        "tokens_per_min": 500,
        "models": [
            "google/gemini-flash-1.5",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.2-3b-instruct:free",
        ],
    },
    "demo-pro": {
        "tier": "pro",
        "tokens_per_min": 20000,
        "models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "google/gemini-flash-1.5",
        ],
    },
    "demo-enterprise": {
        "tier": "enterprise",
        "tokens_per_min": 100000,
        "models": [
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
        ],
    },
}


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key header",
        )
    key_data = API_KEYS.get(x_api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
        )
    return {"api_key": x_api_key, **key_data}
