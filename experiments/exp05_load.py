"""EXP-05 — Concurrency / load lite.

3 rounds at concurrency {2, 5, 10} against PROD /chat/stream with demo-pro key.
First round warms the cache. Records success / 429 / other-fail / p50/p95 latency
and peak active_streams.

Output: experiments/results/exp05_load.csv
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from experiments.common import (
    PROD_URL,
    call_chat_stream,
    eval_factual_comparative,
    load_eval_questions,
    p50,
    p95,
    write_csv,
    write_log,
)

ROUNDS = [2, 5, 10]
REQUESTS_PER_ROUND = 20
API_KEY = "demo-pro"
PER_REQUEST_TIMEOUT = 60.0


async def _sample_health() -> int:
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(f"{PROD_URL}/health")
            if r.status_code == 200:
                return int(r.json().get("active_streams", 0))
    except Exception:
        pass
    return -1


async def _warm_cache(questions: list[dict]) -> None:
    write_log("[exp05] warming cache (sequential)...")
    for q in questions:
        try:
            r = await call_chat_stream(q["question"], api_key=API_KEY, timeout=PER_REQUEST_TIMEOUT)
            write_log(f"[exp05]   warm q={q['id']} status={r['status']} lat={r['latency_ms']}ms")
        except Exception as e:
            write_log(f"[exp05]   warm q={q['id']} error: {e}")


async def _do_round(concurrency: int, questions: list[dict]) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    status_counts: dict[int, int] = {}
    peak_active = 0
    stop_health = asyncio.Event()

    async def health_poller():
        nonlocal peak_active
        while not stop_health.is_set():
            v = await _sample_health()
            if v > peak_active:
                peak_active = v
            try:
                await asyncio.wait_for(stop_health.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

    async def one(q: dict):
        async with sem:
            try:
                r = await call_chat_stream(q["question"], api_key=API_KEY, timeout=PER_REQUEST_TIMEOUT)
                latencies.append(r["latency_ms"])
                status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            except Exception as e:
                status_counts[-1] = status_counts.get(-1, 0) + 1
                write_log(f"[exp05]   request error c={concurrency}: {type(e).__name__}: {str(e)[:120]}")

    poller = asyncio.create_task(health_poller())
    try:
        await asyncio.gather(
            *(one(random.choice(questions)) for _ in range(REQUESTS_PER_ROUND))
        )
    finally:
        stop_health.set()
        await poller

    success = status_counts.get(200, 0)
    fail_429 = status_counts.get(429, 0)
    fail_other = sum(v for k, v in status_counts.items() if k not in (200, 429))
    # -1 counts (connection-level exceptions) are already included via fail_other
    return {
        "concurrent": concurrency,
        "total_requests": REQUESTS_PER_ROUND,
        "success": success,
        "fail_429": fail_429,
        "fail_other": fail_other,
        "p50_ms": round(p50(latencies), 1),
        "p95_ms": round(p95(latencies), 1),
        "peak_active_streams": peak_active,
    }


async def run() -> dict[str, Any]:
    write_log("[exp05] START — load test (concurrency 2/5/10)")
    questions = eval_factual_comparative(load_eval_questions())
    await _warm_cache(questions)

    rows: list[dict] = []
    for c in ROUNDS:
        write_log(f"[exp05] round concurrent={c} starting")
        row = await _do_round(c, questions)
        rows.append(row)
        write_log(
            f"[exp05] round c={c} done: success={row['success']} 429={row['fail_429']} "
            f"other={row['fail_other']} p50={row['p50_ms']}ms p95={row['p95_ms']}ms "
            f"peak_active={row['peak_active_streams']}"
        )
        await asyncio.sleep(2.0)

    write_csv("experiments/results/exp05_load.csv", rows)
    write_log(f"[exp05] END — {len(rows)} rounds saved")
    return {"ok": True, "rows": len(rows)}


if __name__ == "__main__":
    asyncio.run(run())
