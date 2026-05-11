"""Bounded concurrency wrapper around the LLM stream — uses app.state.llm_semaphore."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any


async def call_llm_bounded(
    app_state: Any,
    inner_gen,
) -> AsyncGenerator[dict, None]:
    """Acquire the global LLM semaphore for the duration of an inner async generator."""
    async with app_state.llm_semaphore:
        try:
            async for chunk in inner_gen:
                yield chunk
        finally:
            pass
