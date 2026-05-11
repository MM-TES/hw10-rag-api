"""Async service for inserting RequestLog rows."""
from __future__ import annotations

import hashlib

import structlog

from app.tracking.models import RequestLog

log = structlog.get_logger()


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


async def log_request(
    session_maker,
    *,
    api_key: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    ttft_ms: int | None = None,
    cache_hit: bool = False,
    fallback_used: bool = False,
    output_filtered: bool = False,
) -> None:
    try:
        async with session_maker() as session:
            session.add(
                RequestLog(
                    api_key_hash=hash_api_key(api_key),
                    model=model[:100],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    cache_hit=cache_hit,
                    fallback_used=fallback_used,
                    output_filtered=output_filtered,
                )
            )
            await session.commit()
    except Exception as e:
        log.warning("log_request_failed", error=str(e))
