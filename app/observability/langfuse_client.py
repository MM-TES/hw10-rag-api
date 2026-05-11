"""Langfuse v4 helpers: get_client + start_as_current_observation context managers."""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from typing import Any

import structlog

log = structlog.get_logger()


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@asynccontextmanager
async def trace_chat_request(api_key: str, tier: str, message: str):
    """Wrap a /chat/stream request in a Langfuse span."""
    try:
        from langfuse import get_client
    except Exception:
        yield None
        return
    try:
        lf = get_client()
    except Exception as e:
        log.warning("langfuse_get_client_failed", error=str(e))
        yield None
        return
    try:
        with lf.start_as_current_observation(
            as_type="span",
            name="chat_request",
            input={
                "message": message[:500],
                "api_key_hash": hash_api_key(api_key),
                "tier": tier,
            },
        ) as root:
            yield root
    except Exception as e:
        log.warning("langfuse_trace_failed", error=str(e))
        yield None


def make_span(name: str, input_data: dict | None = None):
    """Return a context manager for a child span (or no-op if Langfuse unavailable)."""
    try:
        from langfuse import get_client
        lf = get_client()
        return lf.start_as_current_observation(as_type="span", name=name, input=input_data or {})
    except Exception:
        from contextlib import nullcontext
        return nullcontext(None)


def make_generation_span(name: str, model: str, input_data: dict | None = None):
    try:
        from langfuse import get_client
        lf = get_client()
        return lf.start_as_current_observation(
            as_type="generation", name=name, model=model, input=input_data or {}
        )
    except Exception:
        from contextlib import nullcontext
        return nullcontext(None)


def safe_update(span: Any, **kwargs) -> None:
    if span is None:
        return
    try:
        span.update(**kwargs)
    except Exception as e:
        log.warning("langfuse_span_update_failed", error=str(e))
