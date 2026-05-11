# Reusable Components from Previous Lessons

Quick reference: which files from `lesson-02` … `lesson-11` you can adapt for HW10. **Don't copy-paste blindly** — review the file, extract the relevant pattern, rewrite for the HW10 context. Most of these were written for batch/offline pipelines, not async API serving.

Paths are relative to the repo root: `C:\Users\MTeslenko\PycharmProjects\ai-engineering_HW\`.

---

## ⭐ lesson-09-rag-systems-enterprise — most relevant

End-to-end RAG benchmarking on MS MARCO, with embedders, retrievers, eval, plotting, and a Claude-based PDF report. The closest analog to HW10 in this repo.

| File | Role | HW10 reuse |
|---|---|---|
| `lesson-09-rag-systems-enterprise/homework/src/embedder.py` | `BGEEmbedder` wrapper around `sentence-transformers` (encode_passages / encode_queries, batching, normalize) | Reference for `app/rag/embedder.py` and `app/deps.py` (singleton). HW10 will use `all-MiniLM-L6-v2` (384-dim) instead of BGE — but the wrapper shape is reusable. |
| `lesson-09-rag-systems-enterprise/homework/src/retrievers.py` | `BaseRetriever` ABC + `DenseFlatRetriever` + `BM25Retriever` | Pattern for `app/rag/retriever.py`. HW10 will plug Qdrant cloud instead of in-memory FAISS, but the ABC + top-k interface stays. |
| `lesson-09-rag-systems-enterprise/homework/src/benchmark.py` | `BenchmarkResult` dataclass, latency percentiles, RAM/disk metrics | Phase 2: adapt for `experiments/common.py` latency aggregations (p50/p95/p99). |
| `lesson-09-rag-systems-enterprise/homework/template/data_loader.py` | Streaming MS MARCO loader (lazy JSON line reader) | Pattern for `scripts/index.py` — streaming over `data/source.md` chunks rather than loading full document. |
| `lesson-09-rag-systems-enterprise/homework/template/metrics.py` | `recall_at_k`, `mrr_at_k` evaluation functions | Phase 2 EXP-01 (chunk-size sweep): use these to measure retrieval quality without LLM. |
| `lesson-09-rag-systems-enterprise/homework/scripts/02_embed.py` | Embedding a corpus end-to-end with progress | Pattern for `scripts/index.py` (corpus → chunks → embed → upsert). |
| `lesson-09-rag-systems-enterprise/homework/scripts/03_retriever_smoketest.py` | Quick smoke-test retriever pipeline | Pattern for `verify_smoke.py` retrieval check. |
| `lesson-09-rag-systems-enterprise/homework/scripts/05_plot_baseline.py` | matplotlib plotting baseline metrics | Phase 2 report figures (`exp01..exp08` plots). |
| `lesson-09-rag-systems-enterprise/homework/scripts/07_plot_compare.py` | Multi-variant comparison plots | Phase 2 EXP-04 model comparison plot. |
| `lesson-09-rag-systems-enterprise/homework/scripts/09_generate_pdf.py` | Jinja2 + WeasyPrint → PDF report | **Direct template** for `report/generate.py` — same pipeline (load CSVs → render figures → render Jinja → WeasyPrint). |
| `lesson-09-rag-systems-enterprise/homework/scripts/10_chart_interpretations.py` | Claude Opus chart-by-chart interpretations | **Direct template** for `report/interpret.py` Opus calls. |

---

## lesson-08-vector-databases-in-production

Benchmark of FAISS / Qdrant / pgvector / Chroma on 523K passages. Heavy `archive/` folder (legacy benchmarks) — focus on the active files.

| File | Role | HW10 reuse |
|---|---|---|
| `lesson-08-vector-databases-in-production/homework/src/metrics.py` | `recall_at_k`, `mrr_at_k`, `evaluate()` | Same as lesson-09 metrics — pick whichever is cleaner. |
| `lesson-08-vector-databases-in-production/homework/src/data_loader.py` | Streaming dataset loader | Reference for memory-aware loading in `scripts/index.py`. |
| `lesson-08-vector-databases-in-production/homework/generate_pdf.py` | Earlier-version PDF generator (Jinja+WeasyPrint) | Cross-check with lesson-09 version when assembling `report/generate.py`. |
| `lesson-08-vector-databases-in-production/homework/CLAUDE.md` | Existing CLAUDE.md style in this repo | Style reference (Ukrainian, terse, "do not modify SPEC" pattern). |

---

## lesson-06-llm-engineering — LLM-as-judge patterns

Closest precedent for evaluation via LLM (HW10 Phase 2 uses Claude Haiku as judge for 600+ scoring calls).

| File | Role | HW10 reuse |
|---|---|---|
| `lesson-06-llm-engineering/homework/judge.py` | LLM-as-judge prompt + scoring loop with Claude | **Direct template** for `experiments/common.py` `judge_answer()` function. Phase 2 EXP-04 uses this exact pattern. |
| `lesson-06-llm-engineering/homework/evaluate.py` | Outer evaluation loop (per-example judging + aggregation) | Pattern for Phase 2 experiment runners (loop over questions × variants). |
| `lesson-06-llm-engineering/homework/generate_analysis.py` | Claude-based summary/analysis generation | Pattern for `report/interpret.py` Opus calls (executive summary, cross-experiment insights). |
| `lesson-06-llm-engineering/homework/charts_generator.py` | matplotlib charts for evaluation results | Mid-complexity plotting reference. |
| `lesson-06-llm-engineering/homework/run_all.py` | Orchestration of full eval pipeline | Pattern for `experiments/run_all.py` — sequential experiment driver with logging. |
| `lesson-06-llm-engineering/CLAUDE.md` | More detailed CLAUDE.md style | Cross-reference when revisiting `./CLAUDE.md`. |

---

## lesson-02-data-engineering, lesson-07-embeddings-semantic-systems, lesson-11-ai-agents-tool-orchestration

Limited reuse for HW10 (different problem shape).

- **lesson-02-data-engineering/homework/evaluate.py** — small evaluation harness, mostly superseded by lesson-06 `evaluate.py`. Skip unless you need a minimal baseline.
- **lesson-07-embeddings-semantic-systems/embedding-failures-lab/app.py** — Streamlit demo of embedding failure modes. Useful only as a teaching reference, not for production code.
- **lesson-11-ai-agents-tool-orchestration/demo/refund-triage/app.py** — FastAPI agent app. Different problem (multi-agent triage, not Q&A bot), but the FastAPI scaffolding can be cross-checked.
- **lesson-11-ai-agents-tool-orchestration/demo/supply-chain/src/judge.py** — alternative LLM-as-judge implementation; cross-reference with lesson-06 `judge.py`.

---

## lesson-03-production-data-pipelines, lesson-04, lesson-05

Skipped — homework folders are mostly empty or off-topic for a Q&A RAG service.

---

## How to use this index

When you need a pattern in HW10, **don't reimplement from scratch** — open the referenced file, study the pattern, and adapt. Examples:

- Building `app/rag/retriever.py`? → Read `lesson-09/.../src/retrievers.py` first.
- Writing `experiments/common.py` `judge_answer()`? → Read `lesson-06/homework/judge.py` first.
- Writing `report/generate.py`? → Read `lesson-09/homework/scripts/09_generate_pdf.py` first.

If a pattern doesn't fit (e.g. lesson-09 retrievers are sync + batch, HW10 needs async), document the deviation in code comments only when the divergence is non-obvious — otherwise the diff itself is self-explanatory.
