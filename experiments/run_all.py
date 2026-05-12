"""Run all 8 Phase 2 experiments in order, with budget checkpoints between them."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from experiments.budget_guard import guard
from experiments.common import write_log
from experiments import (
    exp01_chunking,
    exp02_topk,
    exp03_cache_threshold,
    exp04_models,
    exp05_load,
    exp06_fallback_observed,
    exp07_injection,
    exp08_cost_projection,
    exp09_judge_agreement,
)


EXPERIMENTS = [
    ("EXP-01", exp01_chunking.run, "experiments/results/exp01_chunking.csv"),
    ("EXP-02", exp02_topk.run, "experiments/results/exp02_topk.csv"),
    ("EXP-03", exp03_cache_threshold.run, "experiments/results/exp03_cache_threshold.csv"),
    ("EXP-04", exp04_models.run, "experiments/results/exp04_models.csv"),
    ("EXP-05", exp05_load.run, "experiments/results/exp05_load.csv"),
    ("EXP-06", exp06_fallback_observed.run, "experiments/results/exp06_fallback_observed.csv"),
    ("EXP-07", exp07_injection.run, "experiments/results/exp07_injection.csv"),
    ("EXP-08", exp08_cost_projection.run, "experiments/results/exp08_cost_projection.csv"),
    ("EXP-09", exp09_judge_agreement.run, "experiments/results/exp09_judge_agreement.csv"),
]


async def main() -> int:
    started = time.perf_counter()
    await guard.set_baseline()
    baseline = guard._baseline
    write_log(f"[run_all] START — budget baseline=${baseline:.4f}, cap=${guard.hard_stop:.2f}")
    summary: list[dict] = []

    for name, runner, csv_path in EXPERIMENTS:
        write_log(f"[run_all] === {name} starting ===")
        t0 = time.perf_counter()
        try:
            result = await runner()
        except Exception as e:
            write_log(f"[run_all] {name} CRASHED: {type(e).__name__}: {e}")
            result = {"ok": False, "crash": str(e)[:200]}
        elapsed = round(time.perf_counter() - t0, 1)

        csv_lines = 0
        csv_p = Path(csv_path)
        if csv_p.exists():
            csv_lines = max(0, sum(1 for _ in csv_p.open(encoding="utf-8")) - 1)
        spent = await guard.spent_since_baseline()
        remaining = await guard.remaining()

        summary.append(
            {
                "name": name,
                "elapsed_s": elapsed,
                "result": result,
                "csv_rows": csv_lines,
                "spent_so_far": round(spent, 4),
                "remaining": round(remaining, 4),
            }
        )
        write_log(
            f"[run_all] {name} done in {elapsed}s | csv_rows={csv_lines} | "
            f"spent=${spent:.4f} | remaining=${remaining:.4f}"
        )

        if csv_lines == 0 and result.get("ok", False) is False:
            write_log(f"[run_all] {name} produced empty CSV and failed — but continuing per spec (analytical exps may legitimately be 0)")

    total = round(time.perf_counter() - started, 1)
    final_spent = await guard.spent_since_baseline()
    write_log(f"[run_all] DONE in {total}s | total Phase-2 spend=${final_spent:.4f}")
    write_log(f"[run_all] summary: {summary}")
    await guard.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
