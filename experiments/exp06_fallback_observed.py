"""EXP-06 — Fallback observed.

Analytical (no LLM calls). Queries prod Postgres request_logs to summarize:
  model, n, avg_latency_ms, ttft_p50, fallback_count, fallback_rate, cache_hit_rate, total_cost
Grouped by model.

Output: experiments/results/exp06_fallback_observed.csv
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from experiments.common import write_csv, write_log

load_dotenv()

QUERY = """
SELECT
  model,
  COUNT(*) AS n,
  COALESCE(AVG(latency_ms), 0)::float AS avg_latency_ms,
  COALESCE(AVG(ttft_ms), 0)::float AS avg_ttft_ms,
  SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END) AS fallback_count,
  SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hit_count,
  COALESCE(SUM(cost_usd), 0)::float AS total_cost_usd
FROM request_logs
GROUP BY model
ORDER BY n DESC
"""


async def run() -> dict[str, Any]:
    write_log("[exp06] START — fallback observed (DB query)")
    db_url = os.environ["DATABASE_URL"].replace("?pgbouncer=true", "")
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(db_url, connect_args={"statement_cache_size": 0})

    rows: list[dict] = []
    try:
        async with engine.connect() as conn:
            r = await conn.execute(text(QUERY))
            for row in r.mappings().all():
                n = int(row["n"])
                fb_count = int(row["fallback_count"])
                ch_count = int(row["cache_hit_count"])
                rows.append(
                    {
                        "model": row["model"] or "(unknown)",
                        "n_requests": n,
                        "avg_latency_ms": round(float(row["avg_latency_ms"]), 1),
                        "avg_ttft_ms": round(float(row["avg_ttft_ms"]), 1),
                        "fallback_count": fb_count,
                        "fallback_rate": round(fb_count / n, 4) if n else 0.0,
                        "cache_hit_count": ch_count,
                        "cache_hit_rate": round(ch_count / n, 4) if n else 0.0,
                        "total_cost_usd": round(float(row["total_cost_usd"]), 6),
                    }
                )
                write_log(
                    f"[exp06] {row['model']}: n={n} fb={fb_count} "
                    f"cache={ch_count} avg_lat={row['avg_latency_ms']:.0f}ms"
                )
    finally:
        await engine.dispose()

    write_csv("experiments/results/exp06_fallback_observed.csv", rows)
    write_log(f"[exp06] END — {len(rows)} rows saved")
    return {"ok": True, "rows": len(rows)}


if __name__ == "__main__":
    asyncio.run(run())
