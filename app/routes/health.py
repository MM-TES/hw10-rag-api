"""GET /health — backend connectivity check + active stream counters."""
from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request
from sqlalchemy import text

log = structlog.get_logger()

router = APIRouter()


async def _ping_redis(redis) -> str:
    try:
        ok = await asyncio.wait_for(redis.ping(), timeout=2.0)
        return "ok" if ok else "error"
    except Exception as e:
        log.warning("health_redis_failed", error=str(e))
        return "error"


async def _ping_qdrant(qdrant) -> str:
    try:
        await asyncio.wait_for(asyncio.to_thread(qdrant.get_collections), timeout=2.0)
        return "ok"
    except Exception as e:
        log.warning("health_qdrant_failed", error=str(e))
        return "error"


async def _ping_db(engine) -> str:
    try:
        async def _q():
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        await asyncio.wait_for(_q(), timeout=2.0)
        return "ok"
    except Exception as e:
        log.warning("health_db_failed", error=str(e))
        return "error"


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state
    redis_status, qdrant_status, db_status = await asyncio.gather(
        _ping_redis(state.redis),
        _ping_qdrant(state.qdrant),
        _ping_db(state.db_engine),
    )
    overall = "ok" if all(s == "ok" for s in (redis_status, qdrant_status, db_status)) else "degraded"
    return {
        "status": overall,
        "active_streams": state.active_streams,
        "aborted_streams": state.aborted_streams,
        "redis": redis_status,
        "qdrant": qdrant_status,
        "db": db_status,
    }
