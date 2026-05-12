"""Semantic cache backed by Qdrant cache_collection (expire_at filter for TTL)."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog
from qdrant_client.models import FieldCondition, Filter, PointStruct, Range

from app.config import settings

log = structlog.get_logger()


async def cache_lookup(
    qdrant: Any,
    embedding: list[float],
    threshold: float | None = None,
) -> dict | None:
    """Return cached payload {response, model, query, created_at} on similarity HIT, else None."""
    thr = threshold if threshold is not None else settings.CACHE_SIMILARITY_THRESHOLD
    now = int(time.time())
    response = await asyncio.to_thread(
        qdrant.query_points,
        collection_name=settings.QDRANT_CACHE_COLLECTION,
        query=embedding,
        limit=1,
        score_threshold=thr,
        query_filter=Filter(must=[FieldCondition(key="expire_at", range=Range(gte=now))]),
        with_payload=True,
    )
    if not response.points:
        return None
    top = response.points[0]
    if float(top.score) < thr:
        return None
    return top.payload or None


async def cache_store(
    qdrant: Any,
    embedding: list[float],
    query: str,
    response_text: str,
    model: str,
    sources: list[str],
    original_prompt_tokens: int = 0,
    original_completion_tokens: int = 0,
    original_cost_usd: float = 0.0,
    ttl_seconds: int | None = None,
) -> None:
    ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
    now = int(time.time())
    point = PointStruct(
        id=uuid.uuid4().hex,
        vector=embedding,
        payload={
            "query": query,
            "response": response_text,
            "model": model,
            "sources": sources,
            "expire_at": now + ttl,
            "created_at": now,
            "original_prompt_tokens": int(original_prompt_tokens),
            "original_completion_tokens": int(original_completion_tokens),
            "original_cost_usd": float(original_cost_usd),
        },
    )
    try:
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=settings.QDRANT_CACHE_COLLECTION,
            points=[point],
        )
    except Exception as e:
        log.warning("cache_store_failed", error=str(e))
