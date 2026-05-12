"""EXP-09 — Inter-rater agreement between Haiku 4.5 and gpt-4o-mini judges.

Reads experiments/results/exp{01,02,04}_raw.csv (per-question dual-judge rows)
and computes:
  - Spearman ρ (rank correlation)
  - Kendall τ (alternative rank correlation)
  - Pearson r (linear)
  - mean absolute difference (|haiku - 4o-mini|)
per (experiment × metric) and an overall row.

No new LLM calls. Output: experiments/results/exp09_judge_agreement.csv
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.common import write_csv, write_log

RAW_FILES = ["exp01_raw.csv", "exp02_raw.csv", "exp04_raw.csv"]
METRICS = [("faithfulness", "f_haiku", "f_4omini"),
           ("relevance", "r_haiku", "r_4omini"),
           ("completeness", "c_haiku", "c_4omini")]


def _read_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pairs(rows: list[dict], col_h: str, col_o: str) -> tuple[list[float], list[float]]:
    h, o = [], []
    for r in rows:
        try:
            a = float(r[col_h])
            b = float(r[col_o])
        except (KeyError, ValueError):
            continue
        if a <= 0 or b <= 0:
            # skip judge-unavailable rows
            continue
        h.append(a)
        o.append(b)
    return h, o


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        n = len(vals)
        idx_sorted = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        # average ties
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[idx_sorted[j + 1]] == vals[idx_sorted[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[idx_sorted[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    return _pearson(rx, ry)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tie_x += 1
            elif b == 0:
                tie_y += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + tie_x) * (concordant + discordant + tie_y)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _mad(x: list[float], y: list[float]) -> float:
    if not x:
        return 0.0
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


async def run() -> dict[str, Any]:
    write_log("[exp09] START — judge agreement (no API calls)")
    results_dir = Path("experiments/results")

    all_rows: list[dict] = []
    overall_pairs = {metric: ([], []) for metric, _, _ in METRICS}

    for raw_file in RAW_FILES:
        path = results_dir / raw_file
        rows = _read_raw(path)
        if not rows:
            write_log(f"[exp09] skip {raw_file} — missing or empty")
            continue
        exp_id = raw_file.split("_")[0]  # exp01 / exp02 / exp04

        for metric, col_h, col_o in METRICS:
            h, o = _pairs(rows, col_h, col_o)
            if not h:
                continue
            overall_pairs[metric][0].extend(h)
            overall_pairs[metric][1].extend(o)
            all_rows.append(
                {
                    "experiment": exp_id,
                    "metric": metric,
                    "n_pairs": len(h),
                    "spearman_rho": round(_spearman(h, o), 4),
                    "kendall_tau": round(_kendall_tau(h, o), 4),
                    "pearson_r": round(_pearson(h, o), 4),
                    "mean_abs_diff": round(_mad(h, o), 4),
                    "haiku_mean": round(mean(h), 3),
                    "4omini_mean": round(mean(o), 3),
                    "agreement_exact": round(
                        sum(1 for a, b in zip(h, o) if a == b) / len(h), 4
                    ),
                }
            )

    # Overall (all experiments stacked per metric)
    for metric, _, _ in METRICS:
        h, o = overall_pairs[metric]
        if not h:
            continue
        all_rows.append(
            {
                "experiment": "ALL",
                "metric": metric,
                "n_pairs": len(h),
                "spearman_rho": round(_spearman(h, o), 4),
                "kendall_tau": round(_kendall_tau(h, o), 4),
                "pearson_r": round(_pearson(h, o), 4),
                "mean_abs_diff": round(_mad(h, o), 4),
                "haiku_mean": round(mean(h), 3),
                "4omini_mean": round(mean(o), 3),
                "agreement_exact": round(
                    sum(1 for a, b in zip(h, o) if a == b) / len(h), 4
                ),
            }
        )

    for r in all_rows:
        write_log(
            f"[exp09] {r['experiment']:>5} {r['metric']:>12} "
            f"n={r['n_pairs']:>3} rho={r['spearman_rho']:+.3f} "
            f"tau={r['kendall_tau']:+.3f} r={r['pearson_r']:+.3f} "
            f"MAD={r['mean_abs_diff']:.3f} agree={r['agreement_exact']:.2f}"
        )

    write_csv("experiments/results/exp09_judge_agreement.csv", all_rows)
    write_log(f"[exp09] END — {len(all_rows)} rows saved")
    return {"ok": True, "rows": len(all_rows)}


if __name__ == "__main__":
    asyncio.run(run())
