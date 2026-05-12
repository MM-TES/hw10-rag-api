"""Shared utilities for Phase 2 experiments.

- HTTP client to the production /chat/stream endpoint
- Direct OpenRouter streaming client (bypasses our service for chunking/model sweeps)
- Local Qdrant client + lazy SentenceTransformer
- CSV writer + simple file logger
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROD_URL = os.environ.get("PROD_URL", "https://hw10-rag-api.fly.dev")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

_embedder = None
_qdrant = None
_openrouter = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        write_log(f"[common] loading embedder {EMBED_MODEL_NAME} ...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME.split("/", 1)[-1])
    return _embedder


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient

        _qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=60,
        )
    return _qdrant


def get_openrouter():
    global _openrouter
    if _openrouter is None:
        from openai import AsyncOpenAI

        _openrouter = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            timeout=60.0,
        )
    return _openrouter


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def p50(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def p95(values: list[float]) -> float:
    return percentile(values, 0.95)


# --- Streaming clients ---------------------------------------------------


async def call_chat_stream(
    message: str,
    api_key: str = "demo-pro",
    timeout: float = 60.0,
) -> dict:
    """POST /chat/stream against the production URL; aggregate token+done events."""
    body = {"message": message}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    content_parts: list[str] = []
    done_event: dict | None = None
    error_event: dict | None = None
    started = time.perf_counter()
    ttft_ms: int | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{PROD_URL}/chat/stream",
            json=body,
            headers=headers,
        ) as response:
            status = response.status_code
            if status != 200:
                # Drain response, return early
                body_text = (await response.aread()).decode("utf-8", errors="replace")
                return {
                    "status": status,
                    "content": "",
                    "done": {},
                    "error": body_text[:400],
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "ttft_ms": None,
                }

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = event.get("type")
                if t == "token":
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - started) * 1000)
                    content_parts.append(event.get("content", ""))
                elif t == "done":
                    done_event = event
                elif t == "error":
                    error_event = event

    return {
        "status": 200,
        "content": "".join(content_parts),
        "done": done_event or {},
        "error": (error_event or {}).get("error"),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "ttft_ms": ttft_ms,
    }


async def call_openrouter_stream(
    model: str,
    messages: list[dict],
    timeout: float = 60.0,
) -> dict:
    """Direct OpenRouter streaming call; returns content + usage + timing."""
    client = get_openrouter()
    started = time.perf_counter()
    ttft_ms: int | None = None
    content_parts: list[str] = []
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error_detail: str | None = None

    try:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.2,
            ),
            timeout=timeout,
        )
        try:
            async for event in stream:
                if event.usage:
                    usage = {
                        "prompt_tokens": event.usage.prompt_tokens,
                        "completion_tokens": event.usage.completion_tokens,
                        "total_tokens": event.usage.total_tokens,
                    }
                for choice in event.choices or []:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if ttft_ms is None:
                            ttft_ms = int((time.perf_counter() - started) * 1000)
                        content_parts.append(content)
        finally:
            try:
                await stream.close()
            except Exception:
                pass
    except asyncio.TimeoutError:
        error_detail = f"timeout_{timeout}s"
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)[:180]}"

    return {
        "model": model,
        "content": "".join(content_parts),
        "usage": usage,
        "ttft_ms": ttft_ms,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error": error_detail,
    }


# --- IO helpers ----------------------------------------------------------


def write_csv(filepath: Path | str, rows: list[dict]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        filepath.write_text("(no rows)\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_log(message: str) -> None:
    log_file = RESULTS_DIR / "phase2_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# --- Qdrant temp-collection helpers --------------------------------------


def qdrant_recreate_collection(name: str, vector_size: int = 384) -> None:
    from qdrant_client.models import Distance, VectorParams

    q = get_qdrant()
    if q.collection_exists(name):
        q.delete_collection(name)
    q.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def qdrant_delete_collection(name: str) -> bool:
    q = get_qdrant()
    try:
        if q.collection_exists(name):
            q.delete_collection(name)
            return True
    except Exception as e:
        write_log(f"[common] failed to delete collection {name}: {e}")
    return False


def load_eval_questions() -> list[dict]:
    return json.loads(
        Path("experiments/eval_questions.json").read_text(encoding="utf-8")
    )


def eval_factual_comparative(qs: list[dict]) -> list[dict]:
    return [q for q in qs if q["category"] in ("factual", "comparative")]
