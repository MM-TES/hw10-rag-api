"""Embedder wrapper around app.state.embedder singleton."""
from __future__ import annotations

import asyncio
from typing import Any


async def embed(embedder: Any, text: str) -> list[float]:
    """Encode a single text via the shared SentenceTransformer (off the event loop)."""
    def _do() -> list[float]:
        vec = embedder.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return vec[0].tolist()
    return await asyncio.to_thread(_do)
