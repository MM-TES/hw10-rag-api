"""Lifespan + DI singletons stored on app.state."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

log = structlog.get_logger()


class UpstashRedis:
    """REST client for Upstash Redis (INCRBY/EXPIRE/GET/TTL via HTTPS)."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.client = httpx.AsyncClient(headers=self.headers, timeout=5.0)

    async def incrby(self, key: str, amount: int) -> int:
        r = await self.client.post(f"{self.url}/incrby/{key}/{amount}")
        r.raise_for_status()
        return int(r.json()["result"])

    async def expire(self, key: str, seconds: int) -> bool:
        r = await self.client.post(f"{self.url}/expire/{key}/{seconds}")
        r.raise_for_status()
        return r.json()["result"] == 1

    async def get(self, key: str) -> str | None:
        r = await self.client.get(f"{self.url}/get/{key}")
        r.raise_for_status()
        return r.json().get("result")

    async def ttl(self, key: str) -> int:
        r = await self.client.get(f"{self.url}/ttl/{key}")
        r.raise_for_status()
        return int(r.json()["result"])

    async def ping(self) -> bool:
        r = await self.client.get(f"{self.url}/ping")
        r.raise_for_status()
        return r.json().get("result") == "PONG"

    async def close(self) -> None:
        await self.client.aclose()


async def _ensure_qdrant_collections(client: AsyncQdrantClient) -> None:
    for name in (settings.QDRANT_CHUNKS_COLLECTION, settings.QDRANT_CACHE_COLLECTION):
        exists = await client.collection_exists(name)
        if not exists:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            log.info("qdrant_collection_created", name=name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("lifespan_start", env=settings.APP_ENV)

    embedder = SentenceTransformer(settings.EMBEDDING_MODEL.split("/", 1)[-1])

    qdrant = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30,
    )
    await _ensure_qdrant_collections(qdrant)

    redis = UpstashRedis(settings.UPSTASH_REDIS_REST_URL, settings.UPSTASH_REDIS_REST_TOKEN)

    db_engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
        echo=False,
    )
    db_session_maker = async_sessionmaker(db_engine, expire_on_commit=False)

    from app.tracking.models import Base
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    openrouter = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
    )

    import os
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
    from langfuse import get_client
    langfuse = get_client()

    app.state.embedder = embedder
    app.state.qdrant = qdrant
    app.state.redis = redis
    app.state.db_engine = db_engine
    app.state.db_session_maker = db_session_maker
    app.state.openrouter = openrouter
    app.state.langfuse = langfuse
    app.state.llm_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)
    app.state.active_streams = 0
    app.state.aborted_streams = 0

    log.info("lifespan_ready")
    try:
        yield
    finally:
        log.info("lifespan_shutdown")
        try:
            langfuse.flush()
        except Exception as e:
            log.warning("langfuse_flush_failed", error=str(e))
        await redis.close()
        await qdrant.close()
        await db_engine.dispose()
        await openrouter.close()
        log.info("lifespan_done")
