"""POST /chat/stream — SSE streaming Q&A pipeline."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.llm.pricing import calc_cost
from app.llm.router import stream_with_fallback
from app.rag.embedder import embed
from app.rag.prompt import build_messages
from app.rag.retriever import retrieve

log = structlog.get_logger()

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    key_data: dict = Depends(verify_api_key),
) -> StreamingResponse:
    state = request.app.state

    started = time.perf_counter()
    query_vec = await embed(state.embedder, body.message)
    chunks = await retrieve(state.qdrant, query_vec)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no chunks indexed — run scripts/index.py",
        )
    messages = build_messages(body.message, chunks)
    sources = [c["chunk_id"] for c in chunks]

    async def event_stream():
        state.active_streams += 1
        try:
            ttft_ms: int | None = None
            full_text = ""
            usage: dict[str, int] = {}
            model_used: str | None = None
            fallback_used = False
            error_detail: str | None = None

            async for chunk in stream_with_fallback(
                state.openrouter, key_data["models"], messages
            ):
                if await request.is_disconnected():
                    state.aborted_streams += 1
                    log.info("client_disconnected", api_key=key_data["api_key"])
                    return
                t = chunk.get("type")
                if t == "token":
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - started) * 1000)
                    yield sse({"type": "token", "content": chunk["content"]})
                elif t == "done":
                    full_text = chunk["full_text"]
                    usage = chunk["usage"]
                    model_used = chunk["model"]
                    fallback_used = chunk["fallback_used"]
                elif t == "error":
                    error_detail = chunk.get("detail", "all_models_failed")

            if error_detail:
                yield sse({"type": "error", "error": error_detail})
                return

            cost = calc_cost(
                model_used or "",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            yield sse(
                {
                    "type": "done",
                    "model": model_used,
                    "fallback_used": fallback_used,
                    "cache_hit": False,
                    "sources": sources,
                    "usage": usage,
                    "cost_usd": round(cost, 8),
                    "latency_ms": latency_ms,
                    "ttft_ms": ttft_ms,
                }
            )
        except asyncio.CancelledError:
            state.aborted_streams += 1
            raise
        finally:
            state.active_streams -= 1

    return StreamingResponse(event_stream(), media_type="text/event-stream")
