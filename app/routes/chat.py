"""POST /chat/stream — full SSE pipeline: injection → cache → rag → llm → log."""
from __future__ import annotations

import asyncio
import json
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.cache.semantic import cache_lookup, cache_store
from app.config import settings
from app.llm.pricing import calc_cost
from app.llm.router import stream_with_fallback
from app.observability.langfuse_client import (
    make_generation_span,
    make_span,
    safe_update,
    trace_chat_request,
)
from app.rag.embedder import embed
from app.rag.prompt import build_messages
from app.rag.retriever import retrieve
from app.ratelimit.token_bucket import check_and_consume
from app.security.injection import check_input, check_output
from app.security.logger import log_suspicious_input, log_suspicious_output
from app.tracking.service import log_request

log = structlog.get_logger()

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    key_data: dict = Depends(verify_api_key),
) -> StreamingResponse:
    state = request.app.state

    if settings.ENABLE_INJECTION_DEFENSE:
        clean, reason = check_input(body.message)
        if not clean:
            log_suspicious_input(key_data["api_key"], reason or "?", body.message)
            log.info("input_blocked", reason=reason)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"input rejected: {reason}",
            )

    allowed, retry_after = await check_and_consume(
        state.redis, key_data["api_key"], 0, key_data["tokens_per_min"]
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )

    started = time.perf_counter()
    with make_span("embed_query", {"chars": len(body.message)}) as embed_span:
        query_vec = await embed(state.embedder, body.message)
        safe_update(embed_span, output={"dim": len(query_vec)})

    cache_payload = None
    if settings.ENABLE_CACHE:
        with make_span("cache_lookup", {"threshold": settings.CACHE_SIMILARITY_THRESHOLD}) as c_span:
            try:
                cache_payload = await cache_lookup(state.qdrant, query_vec)
                safe_update(c_span, output={"hit": cache_payload is not None})
            except Exception as e:
                log.warning("cache_lookup_failed", error=str(e))

    if cache_payload:
        return StreamingResponse(
            _cache_replay_stream(
                request,
                state,
                key_data,
                cache_payload,
                started,
            ),
            media_type="text/event-stream",
        )

    with make_span("vector_search", {"top_k": settings.TOP_K}) as v_span:
        chunks = await retrieve(state.qdrant, query_vec)
        safe_update(v_span, output={"hits": len(chunks)})
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no chunks indexed — run scripts/index.py",
        )
    messages = build_messages(body.message, chunks)
    sources = [c["chunk_id"] for c in chunks]

    return StreamingResponse(
        _llm_stream(
            request,
            state,
            key_data,
            body.message,
            messages,
            sources,
            query_vec,
            started,
        ),
        media_type="text/event-stream",
    )


async def _cache_replay_stream(
    request: Request,
    state,
    key_data: dict,
    payload: dict,
    started: float,
):
    state.active_streams += 1
    try:
        response_text: str = payload.get("response", "")
        model_used: str = payload.get("model", "cache")
        sources: list[str] = payload.get("sources", [])
        original_input = int(payload.get("original_prompt_tokens", 0) or 0)
        original_output = int(payload.get("original_completion_tokens", 0) or 0)
        original_cost = float(payload.get("original_cost_usd", 0.0) or 0.0)
        ttft_ms: int | None = None
        for word in response_text.split(" "):
            if await request.is_disconnected():
                state.aborted_streams += 1
                return
            if ttft_ms is None:
                ttft_ms = int((time.perf_counter() - started) * 1000)
            yield sse({"type": "token", "content": word + " "})
            await asyncio.sleep(0.02)

        latency_ms = int((time.perf_counter() - started) * 1000)
        yield sse(
            {
                "type": "done",
                "model": model_used,
                "fallback_used": False,
                "cache_hit": True,
                "sources": sources,
                "usage": {
                    "prompt_tokens": original_input,
                    "completion_tokens": original_output,
                    "total_tokens": original_input + original_output,
                },
                "cost_usd": round(original_cost, 8),
                "cost_saved_usd": round(original_cost, 8),
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
            }
        )
        await log_request(
            state.db_session_maker,
            api_key=key_data["api_key"],
            model=model_used,
            input_tokens=original_input,
            output_tokens=original_output,
            cost_usd=original_cost,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            cache_hit=True,
        )
    finally:
        state.active_streams -= 1


async def _llm_stream(
    request: Request,
    state,
    key_data: dict,
    user_message: str,
    messages: list[dict],
    sources: list[str],
    query_vec: list[float],
    started: float,
):
    state.active_streams += 1
    try:
        ttft_ms: int | None = None
        full_text = ""
        usage: dict[str, int] = {}
        model_used: str | None = None
        fallback_used = False
        error_detail: str | None = None

        async with state.llm_semaphore:
            with make_generation_span("llm_call", key_data["models"][0], {"models": key_data["models"]}) as gen_span:
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
                safe_update(
                    gen_span,
                    output={"response": full_text[:1000]},
                    usage_details={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                    },
                )

        if error_detail:
            estimated = max(
                100, sum(len(m.get("content", "")) for m in messages) // 4
            )
            try:
                await check_and_consume(
                    state.redis, key_data["api_key"], estimated, key_data["tokens_per_min"]
                )
            except Exception as e:
                log.warning("consume_on_error_failed", error=str(e))
            yield sse({"type": "error", "error": error_detail})
            return

        output_filtered = False
        if settings.ENABLE_INJECTION_DEFENSE and check_output(full_text):
            output_filtered = True
            log_suspicious_output(key_data["api_key"], full_text)

        cost = calc_cost(
            model_used or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        if settings.ENABLE_CACHE and not output_filtered:
            try:
                await cache_store(
                    state.qdrant,
                    query_vec,
                    user_message,
                    full_text,
                    model_used or "",
                    sources,
                    original_prompt_tokens=usage.get("prompt_tokens", 0),
                    original_completion_tokens=usage.get("completion_tokens", 0),
                    original_cost_usd=cost,
                )
            except Exception as e:
                log.warning("cache_store_failed", error=str(e))

        consumed_tokens = usage.get("total_tokens", 0) or max(
            1, (len(full_text) + sum(len(m.get("content", "")) for m in messages)) // 4
        )
        allowed, retry_after = await check_and_consume(
            state.redis,
            key_data["api_key"],
            consumed_tokens,
            key_data["tokens_per_min"],
        )

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
                "output_filtered": output_filtered,
                "rate_limit_exceeded": (not allowed),
                "retry_after": retry_after if not allowed else 0,
            }
        )

        await log_request(
            state.db_session_maker,
            api_key=key_data["api_key"],
            model=model_used or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=cost,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            cache_hit=False,
            fallback_used=fallback_used,
            output_filtered=output_filtered,
        )
    except asyncio.CancelledError:
        state.aborted_streams += 1
        raise
    finally:
        state.active_streams -= 1
