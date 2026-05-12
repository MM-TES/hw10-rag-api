"""EXP-02 — top-K sweep.

Reads exp01_chunking.csv, picks best chunk_size by sum of judge scores,
rebuilds a single temp collection at that size, then sweeps top_k ∈ {1,2,3,5,8}.

Output: experiments/results/exp02_topk.csv
"""
from __future__ import annotations

import asyncio
import csv
import uuid
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm.pricing import calc_cost
from app.rag.prompt import build_messages
from experiments.budget_guard import BudgetExceededError, guard
from experiments.common import (
    call_openrouter_stream,
    eval_factual_comparative,
    get_embedder,
    get_qdrant,
    load_eval_questions,
    qdrant_delete_collection,
    qdrant_recreate_collection,
    write_csv,
    write_log,
)
from experiments.judge import judge_answer

SOURCE = Path("data/source.md")
EVAL_MODEL = "openai/gpt-4o-mini"
TOP_K_VALUES = [1, 2, 3, 5, 8]
PER_CALL_BUDGET_USD = 0.005


def _pick_best_chunk_size() -> int:
    csv_path = Path("experiments/results/exp01_chunking.csv")
    if not csv_path.exists():
        write_log("[exp02] no exp01 csv — defaulting chunk_size=500")
        return 500
    with csv_path.open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr if r.get("chunk_size")]
    if not rows:
        return 500
    def total(r):
        try:
            return float(r["judge_faithfulness_avg"]) + float(r["judge_relevance_avg"]) + float(r["judge_completeness_avg"])
        except Exception:
            return -1.0
    best = max(rows, key=total)
    cs = int(best["chunk_size"])
    write_log(f"[exp02] picked best chunk_size={cs} (sum-score={total(best):.3f})")
    return cs


def _build_collection(name: str, chunks: list[str]) -> int:
    from qdrant_client.models import PointStruct

    embedder = get_embedder()
    qdrant = get_qdrant()
    qdrant_recreate_collection(name, vector_size=384)
    vectors = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v.tolist(),
            payload={"chunk_id": f"c_{i:04d}", "text": c},
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    BATCH = 64
    for i in range(0, len(points), BATCH):
        qdrant.upsert(collection_name=name, points=points[i : i + BATCH])
    return len(points)


async def _retrieve(collection: str, qvec: list[float], k: int) -> list[dict]:
    qdrant = get_qdrant()
    resp = await asyncio.to_thread(
        qdrant.query_points,
        collection_name=collection,
        query=qvec,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": (p.payload or {}).get("chunk_id", str(p.id)),
            "text": (p.payload or {}).get("text", ""),
            "score": float(p.score),
        }
        for p in resp.points
    ]


async def run() -> dict[str, Any]:
    write_log("[exp02] START — top-K sweep")
    if not SOURCE.exists():
        write_log("[exp02] FATAL: data/source.md missing")
        write_csv("experiments/results/exp02_topk.csv", [])
        return {"ok": False}

    chunk_size = _pick_best_chunk_size()
    collection = f"chunks_exp02_{chunk_size}"

    text = SOURCE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(20, int(chunk_size * 0.10)),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    n_chunks = _build_collection(collection, chunks)
    write_log(f"[exp02] indexed {n_chunks} chunks @ cs={chunk_size}")

    embedder = get_embedder()
    eval_qs = eval_factual_comparative(load_eval_questions())

    rows: list[dict] = []
    aborted = False
    # Precompute query vectors once (shared across k values)
    qvecs = {
        q["id"]: embedder.encode([q["question"]], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
        for q in eval_qs
    }

    try:
        for k in TOP_K_VALUES:
            faith, rel, comp = [], [], []
            input_tokens_sum = 0
            cost_sum = 0.0
            calls = 0
            for q in eval_qs:
                try:
                    await guard.check(projected_usd=PER_CALL_BUDGET_USD)
                except BudgetExceededError as e:
                    write_log(f"[exp02] BUDGET HIT at k={k}: {e}")
                    aborted = True
                    break
                retrieved = await _retrieve(collection, qvecs[q["id"]], k)
                messages = build_messages(q["question"], retrieved)
                resp = await call_openrouter_stream(EVAL_MODEL, messages, timeout=45.0)
                if resp["error"]:
                    write_log(f"[exp02]   q={q['id']} k={k} LLM error: {resp['error']}")
                    continue
                cost = calc_cost(
                    EVAL_MODEL,
                    resp["usage"]["prompt_tokens"],
                    resp["usage"]["completion_tokens"],
                )
                cost_sum += cost
                input_tokens_sum += resp["usage"]["prompt_tokens"]
                judged = await judge_answer(
                    q["question"],
                    q.get("expected_keywords", []),
                    resp["content"],
                    category=q["category"],
                )
                faith.append(judged["faithfulness"])
                rel.append(judged["relevance"])
                comp.append(judged["completeness"])
                calls += 1

            if calls > 0:
                rows.append(
                    {
                        "top_k": k,
                        "chunk_size": chunk_size,
                        "n_eval_calls": calls,
                        "judge_faithfulness_avg": round(sum(faith) / len(faith), 3),
                        "judge_relevance_avg": round(sum(rel) / len(rel), 3),
                        "judge_completeness_avg": round(sum(comp) / len(comp), 3),
                        "avg_input_tokens": round(input_tokens_sum / calls, 1),
                        "total_cost_usd": round(cost_sum, 6),
                    }
                )
                write_log(
                    f"[exp02] k={k} done: "
                    f"F={rows[-1]['judge_faithfulness_avg']} "
                    f"R={rows[-1]['judge_relevance_avg']} "
                    f"C={rows[-1]['judge_completeness_avg']} "
                    f"cost=${rows[-1]['total_cost_usd']:.4f}"
                )
            if aborted:
                break
    finally:
        qdrant_delete_collection(collection)
        write_csv("experiments/results/exp02_topk.csv", rows)

    write_log(f"[exp02] END — {len(rows)} rows saved; aborted={aborted}")
    return {"ok": not aborted and bool(rows), "rows": len(rows), "chunk_size": chunk_size}


if __name__ == "__main__":
    asyncio.run(run())
