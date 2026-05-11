"""Token-bucket rate limiting via Upstash Redis REST (INCRBY + EXPIRE, no Lua)."""
from __future__ import annotations

import structlog

log = structlog.get_logger()


async def check_and_consume(
    redis,
    api_key: str,
    tokens_used: int,
    limit_per_min: int,
) -> tuple[bool, int]:
    """Increment per-key counter; if over limit, return (False, retry_after_seconds)."""
    key = f"bucket:{api_key}:tokens_used"
    try:
        used = await redis.incrby(key, tokens_used)
        if tokens_used == 0 and used == 0:
            await redis.expire(key, 60)
    except Exception as e:
        log.warning("ratelimit_redis_failed", error=str(e))
        return True, 0

    if used > limit_per_min:
        try:
            ttl = await redis.ttl(key)
        except Exception:
            ttl = 60
        return False, max(ttl, 1)
    return True, 0
