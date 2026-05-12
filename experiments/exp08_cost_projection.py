"""EXP-08 — Cost projection at scale.

Analytical, $0 cost. Combines:
  - per-model avg cost from exp04_models.csv (with fallbacks if file missing)
  - tier mix {free:0.6, pro:0.3, ent:0.1}
  - cache hit rates {0%, 30%, 60%}
  - volumes {1k, 10k, 100k} req/day

For each tier we use the primary model from app.auth.API_KEYS.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from app.auth import API_KEYS
from app.llm.pricing import PRICING
from experiments.common import write_csv, write_log

VOLUMES = [1_000, 10_000, 100_000]
HIT_RATES = [0.0, 0.30, 0.60]
TIER_MIX = {"free": 0.60, "pro": 0.30, "enterprise": 0.10}


def _model_cost_table() -> dict[str, float]:
    """Map model -> avg cost per request. Prefer exp04_models.csv values."""
    csv_path = Path("experiments/results/exp04_models.csv")
    out: dict[str, float] = {}
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    out[r["model"]] = float(r["cost_per_request_avg"])
                except Exception:
                    pass
    # Fallback estimates: assume ~1500 input + ~150 output tokens per request
    for m, p in PRICING.items():
        if m not in out:
            out[m] = (1500 * p["input"] + 150 * p["output"]) / 1_000_000
    return out


def _primary_model(tier: str) -> str:
    chain = API_KEYS.get(f"demo-{tier}", {}).get("models", [])
    return chain[0] if chain else "openai/gpt-4o-mini"


async def run() -> dict[str, Any]:
    write_log("[exp08] START — cost projection")
    model_cost = _model_cost_table()
    tier_to_model = {t: _primary_model(t) for t in TIER_MIX}
    write_log(f"[exp08] tier->primary_model: {tier_to_model}")

    rows: list[dict] = []
    for volume in VOLUMES:
        for hit in HIT_RATES:
            blended = 0.0
            for tier, share in TIER_MIX.items():
                m = tier_to_model[tier]
                cost = model_cost.get(m, 0.0)
                blended += share * cost
            cost_per_day = volume * blended * (1.0 - hit)
            rows.append(
                {
                    "volume_per_day": volume,
                    "tier_mix": "free60/pro30/ent10",
                    "cache_hit_rate": hit,
                    "blended_cost_per_request": round(blended, 6),
                    "cost_per_day_usd": round(cost_per_day, 4),
                    "cost_per_month_usd": round(cost_per_day * 30, 2),
                }
            )

    write_csv("experiments/results/exp08_cost_projection.csv", rows)
    write_log(f"[exp08] END — {len(rows)} scenarios saved")
    return {"ok": True, "rows": len(rows)}


if __name__ == "__main__":
    asyncio.run(run())
