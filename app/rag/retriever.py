"""Top-k retrieval from Qdrant chunks_collection."""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings


async def retrieve(qdrant: Any, query_vector: list[float], top_k: int | None = None) -> list[dict]:
    """Return list of {chunk_id, text, score, section} for top matches."""
    k = top_k or settings.TOP_K
    response = await asyncio.to_thread(
        qdrant.query_points,
        collection_name=settings.QDRANT_CHUNKS_COLLECTION,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    out: list[dict] = []
    for r in response.points:
        payload = r.payload or {}
        out.append(
            {
                "chunk_id": payload.get("chunk_id", str(r.id)),
                "text": payload.get("text", ""),
                "section": payload.get("section", ""),
                "score": float(r.score),
            }
        )
    return out
