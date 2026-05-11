"""GET /usage/today and /usage/breakdown — per-key cost and latency stats."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import case, func, select

from app.auth import verify_api_key
from app.tracking.models import RequestLog
from app.tracking.service import hash_api_key

router = APIRouter()


@router.get("/usage/today")
async def usage_today(request: Request, key_data: dict = Depends(verify_api_key)) -> dict:
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    h = hash_api_key(key_data["api_key"])
    state = request.app.state
    async with state.db_session_maker() as session:
        result = await session.execute(
            select(
                func.coalesce(func.sum(RequestLog.cost_usd), 0).label("cost"),
                func.coalesce(func.sum(RequestLog.input_tokens + RequestLog.output_tokens), 0).label("tokens"),
                func.count().label("requests"),
            )
            .where(RequestLog.api_key_hash == h)
            .where(RequestLog.created_at >= today_utc)
        )
        row = result.one()
    return {
        "api_key_hash": h,
        "tier": key_data["tier"],
        "since": today_utc.isoformat(),
        "requests": int(row.requests),
        "tokens": int(row.tokens),
        "cost_usd": float(row.cost),
    }


@router.get("/usage/breakdown")
async def usage_breakdown(request: Request, key_data: dict = Depends(verify_api_key)) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    h = hash_api_key(key_data["api_key"])
    state = request.app.state
    out: list[dict] = []
    async with state.db_session_maker() as session:
        result = await session.execute(
            select(
                RequestLog.model,
                func.count().label("n"),
                func.sum(RequestLog.cost_usd).label("cost"),
                func.avg(RequestLog.latency_ms).label("avg_latency_ms"),
                func.sum(case((RequestLog.cache_hit.is_(True), 1), else_=0)).label("cache_hits"),
                func.sum(case((RequestLog.fallback_used.is_(True), 1), else_=0)).label("fallbacks"),
            )
            .where(RequestLog.api_key_hash == h)
            .where(RequestLog.created_at >= since)
            .group_by(RequestLog.model)
        )
        for r in result.all():
            n = int(r.n) or 1
            out.append(
                {
                    "model": r.model,
                    "requests": int(r.n),
                    "cost_usd": float(r.cost or 0),
                    "avg_latency_ms": int(r.avg_latency_ms or 0),
                    "cache_hit_rate": float(r.cache_hits or 0) / n,
                    "fallback_rate": float(r.fallbacks or 0) / n,
                }
            )
    return {"since": since.isoformat(), "by_model": out}
