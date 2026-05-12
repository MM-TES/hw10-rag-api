"""EXP-04 — Model comparison (the expensive one).

Runs 10 factual+comparative questions against 5 models via OpenRouter direct.
If guard.remaining() < 3.0, skips the premium openai/gpt-4o model.
Measures ttft / latency p50/p95, avg judge scores, avg cost.

Output: experiments/results/exp04_models.csv
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
    p50,
    p95,
    qdrant_delete_collection,
    qdrant_recreate_collection,
    write_csv,
    write_log,
)
from experiments.judge import judge_answer

ALL_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-flash-1.5",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
]
PREMIUM_MODEL = "openai/gpt-4o"
TOP_K = 3
PER_CALL_BUDGET_USD = 0.15  # premium worst-case headroom


def _read_best_chunk_size() -> int:
    csv_path = Path("experiments/results/exp01_chunking.csv")
    if not csv_path.exists():
        return 500
    with csv_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("chunk_size")]
    if not rows:
        return 500

    def total(r):
        try:
            return float(r["judge_faithfulness_avg"]) + float(r["judge_relevance_avg"]) + float(r["judge_completeness_avg"])
        except Exception:
            return -1.0

    return int(max(rows, key=total)["chunk_size"])


def _build_collection(name: str, chunk_size: int) -> None:
    from qdrant_client.models import PointStruct

    text = Path("data/source.md").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(20, int(chunk_size * 0.10)),
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

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
    write_log("[exp04] START — model comparison")
    try:
        await guard.check(projected_usd=2.50)
        run_premium = True
    except BudgetExceededError as e:
        write_log(f"[exp04] budget headroom insufficient for premium: {e}")
        run_premium = False
    remaining = await guard.remaining()
    if remaining < 3.0:
        write_log(f"[exp04] remaining=${remaining:.4f} < $3.0 — SKIPPING premium gpt-4o")
        run_premium = False
    models = ALL_MODELS if run_premium else [m for m in ALL_MODELS if m != PREMIUM_MODEL]
    write_log(f"[exp04] will run {len(models)} models: {models}")

    chunk_size = _read_best_chunk_size()
    collection = f"chunks_exp04_{chunk_size}"
    _build_collection(collection, chunk_size)
    write_log(f"[exp04] indexed chunks @ cs={chunk_size} into {collection}")

    embedder = get_embedder()
    eval_qs = eval_factual_comparative(load_eval_questions())
    qvecs = {
        q["id"]: embedder.encode([q["question"]], show_progress_bar=False, convert_to_numpy=True)[0].tolist()
        for q in eval_qs
    }
    retrieved_cache = {q["id"]: await _retrieve(collection, qvecs[q["id"]], TOP_K) for q in eval_qs}

    rows: list[dict] = []
    aborted = False

    try:
        for model in models:
            faith, rel, comp = [], [], []
            ttfts: list[float] = []
            latencies: list[float] = []
            costs: list[float] = []
            errors = 0
            calls = 0

            for q in eval_qs:
                try:
                    await guard.check(projected_usd=PER_CALL_BUDGET_USD)
                except BudgetExceededError as e:
                    write_log(f"[exp04] BUDGET HIT mid-model {model}: {e}")
                    aborted = True
                    break

                messages = build_messages(q["question"], retrieved_cache[q["id"]])
                resp = await call_openrouter_stream(model, messages, timeout=55.0)
                if resp["error"]:
                    errors += 1
                    write_log(f"[exp04]   {model} q={q['id']} err: {resp['error']}")
                    continue
                if resp["ttft_ms"] is not None:
                    ttfts.append(resp["ttft_ms"])
                latencies.append(resp["latency_ms"])
                cost = calc_cost(
                    model,
                    resp["usage"]["prompt_tokens"],
                    resp["usage"]["completion_tokens"],
                )
                costs.append(cost)
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
                        "model": model,
                        "n_requests": calls,
                        "errors": errors,
                        "faithfulness_avg": round(sum(faith) / len(faith), 3),
                        "relevance_avg": round(sum(rel) / len(rel), 3),
                        "completeness_avg": round(sum(comp) / len(comp), 3),
                        "ttft_p50": round(p50(ttfts), 1),
                        "ttft_p95": round(p95(ttfts), 1),
                        "latency_p50": round(p50(latencies), 1),
                        "latency_p95": round(p95(latencies), 1),
                        "cost_per_request_avg": round(sum(costs) / len(costs), 6),
                        "cost_total_usd": round(sum(costs), 6),
                    }
                )
                write_log(
                    f"[exp04] {model}: F/R/C="
                    f"{rows[-1]['faithfulness_avg']}/"
                    f"{rows[-1]['relevance_avg']}/"
                    f"{rows[-1]['completeness_avg']} "
                    f"ttft_p50={rows[-1]['ttft_p50']}ms "
                    f"lat_p50={rows[-1]['latency_p50']}ms "
                    f"cost/req=${rows[-1]['cost_per_request_avg']:.5f}"
                )
            else:
                write_log(f"[exp04] {model}: no successful calls (errors={errors})")
            if aborted:
                break
    finally:
        qdrant_delete_collection(collection)
        write_csv("experiments/results/exp04_models.csv", rows)

    write_log(f"[exp04] END — {len(rows)} models reported; aborted={aborted}")
    return {
        "ok": bool(rows),
        "rows": len(rows),
        "ran_premium": run_premium,
        "models_run": [r["model"] for r in rows],
    }


if __name__ == "__main__":
    asyncio.run(run())
