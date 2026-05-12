"""EXP-03 — cache similarity threshold trade-off (LOCAL only, $0).

For each threshold in {0.80, 0.85, 0.90, 0.92, 0.95, 0.98}, classify pairwise
cosine similarities of paraphrase vs unrelated phrasings:
  - TP: same paraphrase group, sim >= threshold
  - FP: different group (incl. negatives), sim >= threshold
  - FN: same group, sim < threshold
  - TN: different group, sim < threshold
Compute TPR = TP/(TP+FN), FPR = FP/(FP+TN).
"""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from experiments.common import (
    get_embedder,
    load_eval_questions,
    write_csv,
    write_log,
)

THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    n = vecs / np.maximum(norm, 1e-12)
    return n @ n.T


async def run() -> dict[str, Any]:
    write_log("[exp03] START — cache threshold sweep (local-only)")
    qs = load_eval_questions()
    para_qs = [q for q in qs if q["category"] == "paraphrase"]
    factual_qs = [q for q in qs if q["category"] == "factual"][:5]

    items: list[tuple[str, str]] = []  # (group_label, phrasing)
    for q in para_qs:
        group = q["id"]
        items.append((group, q["question"]))
        for p in q.get("paraphrases", []):
            items.append((group, p))
    for q in factual_qs:
        items.append((f"NEG_{q['id']}", q["question"]))

    texts = [t for _, t in items]
    labels = [g for g, _ in items]

    embedder = get_embedder()
    vecs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    sim = _cosine_matrix(vecs)

    pairs_same: list[float] = []
    pairs_diff: list[float] = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if labels[i] == labels[j]:
                pairs_same.append(s)
            else:
                pairs_diff.append(s)
    write_log(
        f"[exp03] pairs: same={len(pairs_same)} diff={len(pairs_diff)} "
        f"(items={n})"
    )

    rows: list[dict] = []
    for thr in THRESHOLDS:
        tp = sum(1 for s in pairs_same if s >= thr)
        fn = sum(1 for s in pairs_same if s < thr)
        fp = sum(1 for s in pairs_diff if s >= thr)
        tn = sum(1 for s in pairs_diff if s < thr)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append(
            {
                "threshold": thr,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
            }
        )
        write_log(f"[exp03] thr={thr}: TP={tp} FP={fp} FN={fn} TN={tn} TPR={tpr:.3f} FPR={fpr:.3f}")

    write_csv("experiments/results/exp03_cache_threshold.csv", rows)
    write_log(f"[exp03] END — {len(rows)} rows saved")
    return {"ok": True, "rows": len(rows)}


if __name__ == "__main__":
    asyncio.run(run())
