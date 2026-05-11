"""LLM router with fallback chain (timeout, 429/5xx → next model)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from openai import APIError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import settings
from app.llm.circuit_breaker import breaker

log = structlog.get_logger()


async def _stream_one_model(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
) -> AsyncGenerator[dict, None]:
    """Stream tokens from a single model. Yields token dicts, then a final usage dict."""
    full_text_parts: list[str] = []
    usage_data: dict | None = None
    stream = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.2,
        ),
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    try:
        async for event in stream:
            if event.usage:
                usage_data = {
                    "prompt_tokens": event.usage.prompt_tokens,
                    "completion_tokens": event.usage.completion_tokens,
                    "total_tokens": event.usage.total_tokens,
                }
            for choice in event.choices or []:
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    full_text_parts.append(content)
                    yield {"type": "token", "content": content}
    finally:
        try:
            await stream.close()
        except Exception:
            pass

    yield {
        "type": "_finish",
        "model": model,
        "full_text": "".join(full_text_parts),
        "usage": usage_data or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def stream_with_fallback(
    client: AsyncOpenAI,
    models: list[str],
    messages: list[dict],
) -> AsyncGenerator[dict, None]:
    """Try each model in turn; fall back on 429/5xx/timeout/network."""
    fallback_used = False
    last_error: str | None = None
    chosen_model: str | None = None

    for i, model in enumerate(models):
        if breaker.is_open(model):
            log.info("breaker_open_skip", model=model)
            last_error = f"breaker_open:{model}"
            continue
        try:
            full_text = ""
            usage: dict[str, int] = {}
            async for chunk in _stream_one_model(client, model, messages):
                if chunk.get("type") == "token":
                    yield chunk
                elif chunk.get("type") == "_finish":
                    full_text = chunk["full_text"]
                    usage = chunk["usage"]
                    chosen_model = chunk["model"]
            breaker.reset(model)
            yield {
                "type": "done",
                "model": chosen_model or model,
                "fallback_used": fallback_used,
                "usage": usage,
                "full_text": full_text,
            }
            return
        except APITimeoutError as e:
            breaker.record_failure(model)
            last_error = f"timeout: {e}"
            log.warning("llm_timeout", model=model)
        except APIStatusError as e:
            breaker.record_failure(model)
            last_error = f"http_{e.status_code}: {str(e)[:200]}"
            log.warning("llm_status_error", model=model, status=e.status_code)
            if e.status_code < 400 or (400 <= e.status_code < 429 and e.status_code != 408):
                pass
        except APIError as e:
            breaker.record_failure(model)
            last_error = f"api_error: {str(e)[:200]}"
            log.warning("llm_api_error", model=model)
        except (asyncio.TimeoutError, Exception) as e:
            breaker.record_failure(model)
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("llm_unexpected", model=model, error=last_error)

        fallback_used = True

    yield {
        "type": "error",
        "error": "all_models_failed",
        "detail": last_error,
        "models_tried": models,
    }
