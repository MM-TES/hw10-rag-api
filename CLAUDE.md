# HW10 — Production RAG API · Project Context

## Mission

Q&A bot про **The Twelve-Factor App** (https://12factor.net) з повним production stack: FastAPI + SSE streaming, semantic cache, token-based rate limiting, multi-provider fallback (OpenRouter), cost tracking, prompt-injection defense, Langfuse observability, опційний Fly.io deploy.

- **Phase 1** (CLAUDE_CODE_PHASE1.md) — будуєш сервіс. Acceptance: `verify_smoke.py` 100% green.
- **Phase 2** (CLAUDE_CODE_PHASE2.md) — нічна автономна сесія: 8 експериментів + PDF звіт під $10 budget.

---

## Read Order (КРИТИЧНО для Claude Code)

1. **ТЗ:** `../README.md` — immutable, source of truth для всіх acceptance criteria.
2. **Поточна фаза:**
   - Setup готовий, Phase 1 ще не commit'нутий → читай `../_tasks/CLAUDE_CODE_PHASE1.md`.
   - `Phase 1 complete: service operational` уже в `git log` → читай `../_tasks/CLAUDE_CODE_PHASE2.md`.
3. **Setup (одноразово):** `./SETUP.md` (короткий launcher) → `../_tasks/SETUP_CHECKLIST.md` (детальний human-walkthrough).
4. **Reuse:** `./REUSABLE_FROM_PREVIOUS_LESSONS.md` — що тягнути з lesson-06..09.

---

## Source of Truth (хто власник чого)

| Артефакт | Owner |
|---|---|
| Endpoints contract, acceptance criteria | `../README.md §Технічні вимоги` |
| `pyproject.toml` версії пакетів | `../_tasks/CLAUDE_CODE_PHASE1.md §Крок 2` |
| `.env` schema | `./.env.example` + `../_tasks/SETUP_CHECKLIST.md §Скласти весь .env` |
| Build order файлів `app/` | `../_tasks/CLAUDE_CODE_PHASE1.md §Крок 3` (17 пунктів, суворе ordering) |
| Experiments + budget guard | `../_tasks/CLAUDE_CODE_PHASE2.md` |
| Eval dataset (20 questions × 4 categories) | `../_tasks/CLAUDE_CODE_PHASE2.md §Крок 1` |

**Якщо щось у цих джерелах суперечить — питай користувача, не вибирай мовчки.**

---

## Architecture (request flow)

```
                         ┌─────────────────────────────────────────────┐
client ── POST /chat/stream ─►│ auth → ratelimit → embed query ──┐         │
                              │                                   ▼         │
                              │        ┌─── cache hit (Qdrant) ──► stream replay
                              │        │                          │         │
                              │   ┌────┴────┐                     │         │
                              │   │ MISS:   │                     │         │
                              │   │ retrieve top-3 ─► build prompt ─► LLM router (OpenRouter)
                              │   │                                    │ primary → fallback1 → fallback2
                              │   │                                    │ circuit breaker (5 fails / 60s)
                              │   │                                    ▼
                              │   │                              SSE tokens ── stream ── client
                              │   │                                    │
                              │   └─► cache upsert ◄── post-stream ────┤
                              │                                        │
                              └────── tracking (Postgres) ◄────────────┘
                                      Langfuse trace (full pipeline)
                                      ratelimit deduct (real tokens)
```

Backends: **Qdrant Cloud** (chunks_collection + cache_collection), **Upstash Redis** (rate-limit counters, REST API only), **Supabase Postgres** (request_logs cost table), **Langfuse Cloud** (traces+spans), **OpenRouter** (LLM gateway).

---

## Key Commands (PowerShell)

```powershell
# One-time setup (from this directory):
conda create -n hw10 python=3.11 -y
conda activate hw10
pip install -e ".[dev]"
Copy-Item .env.example .env
# ... fill .env per ../_tasks/SETUP_CHECKLIST.md ...

# Validate setup:
python scripts\verify_all.py        # must be 7/7 green (or 6/7 with Fly skipped)

# Phase 1 workflow:
python scripts\download_doc.py      # clone 12factor → data/source.md
python scripts\index.py             # chunk + embed + upsert to Qdrant
uvicorn app.main:app --reload       # local server on :8000
python verify_smoke.py              # Phase 1 acceptance test

# Phase 2 workflow (lights-out):
python experiments\preflight.py
python experiments\run_all.py       # honors budget_guard.py
python report\generate.py
```

---

## Phase Guard

- **In Phase 1:** будуєш `app/`, `scripts/`. **НЕ** створюй `experiments/`, `report/`, `eval_questions.json`. Не запускай навантажувальні тести з реальними LLM.
- **In Phase 2:** будуєш `experiments/`, `report/`. **НЕ** модифікуй `app/` (Phase 1 артефакти immutable). Кожен LLM-виклик проходить через `budget_guard.check()`.
- **Перехід Phase 1 → Phase 2:** `git log --oneline | head -1` має містити `Phase 1 complete: service operational` AND `verify_smoke.py` 100% green.

---

## Critical Constraints

- **НЕ модифікуй** `../README.md` чи `../_tasks/*.md` — це ТЗ, immutable. Знайшов помилку → задокументуй у звіті, не правь.
- **НЕ комітти** `.env`, `data/source.md`, `logs/`, `experiments/results/`, `report/output/` (вже у `.gitignore`).
- **Бюджет: $10 OpenRouter physical cap, $9.50 BudgetGuard hard_stop.** Phase 2 — кожен експеримент і кожен LLM-виклик перед запуском викликає `await guard.check()`. Перевищення → graceful stop, write `STOPPED.txt`, exit 0.
- **Judge model:** Claude **Haiku** 4.5 (`claude-haiku-4-5-20251001`), не Opus. Opus = $20+ для ~600 викликів суддівства, Haiku = ~$1.
- **НЕ використовуй** `LangChain RetrievalQA`, `LlamaIndex QueryEngine` чи інші high-level RAG abstractions — пишемо RAG руками. Дозволено: OpenAI/Anthropic SDK, vector DB clients, `langchain-text-splitters` (окремий пакет), `sentence-transformers`, `pypdf`, `tiktoken`.
- **НЕ деплой** на Fly.io автоматично — тільки за прямою інструкцією користувача.

---

## Critical ENV Vars (повний список — у `.env.example`)

- `OPENROUTER_API_KEY` — фінансовий cap, перевірений у Setup. Один ключ = 200+ моделей.
- `DATABASE_URL` — мусить бути `postgresql+asyncpg://...:6543/...` (Supabase pooler + asyncpg). Direct connection (5432) ламає Fly.io через IPv6.
- `EXPERIMENT_HARD_STOP_USD=9.50` — abort condition в Phase 2. Не послаблюй.
- `UPSTASH_REDIS_REST_URL` / `_TOKEN` — Upstash REST API, **без Lua scripts**, тільки `INCRBY`+`EXPIRE` patterns.

---

## Code Conventions (короткий звід)

- **Async** усе I/O (`asyncpg`, `httpx`, `openai.AsyncOpenAI`, `anthropic.AsyncAnthropic`).
- **Type hints** обовʼязково. Pydantic models для всіх API requests/responses.
- **Logging:** `structlog.get_logger().info(..., key=value)`, **ніяких** `print()` в `app/` коді (можна в `scripts/verify_*.py`).
- **Imports:** stdlib → third-party → local (`app.*`). Heavy imports (`sentence_transformers`) — lazy в `app/deps.py` (одиничний singleton).
- **Pathlib** замість `os.path.join`. Helper functions ≤20 рядків.
- **Secrets** тільки через `app.config.settings` (Pydantic Settings). Жодного `os.environ[...]` поза `config.py`.

Повний стиль-гайд: `../_tasks/CLAUDE_CODE_PHASE1.md §Стиль коду`.

---

## When Things Break

1. Спробуй 1 раз інший підхід.
2. Якщо знову не виходить → **STOP**, виведи (що пробував / помилка / варіант обходу) і чекай користувача. Не "обходь" мовчки.

Особливі грабельники (з PHASE1.md §Якщо щось не виходить):

- **WeasyPrint на Windows:** потребує GTK runtime. Не намагайся встановити автоматично — лиши HTML як fallback deliverable.
- **sentence-transformers SSL на Windows:** іноді падає при першому download моделі. Workaround у README, не вирішуй сам.
- **Upstash Redis:** **немає Lua**. `EXPIRE` після `INCR` не атомарно — використовуй `INCRBY` із seed-значенням або pipeline через REST.
- **Qdrant:** **немає built-in TTL**. Зберігай `expire_at` (Unix timestamp) у payload, фільтруй у scroll/search.

---

## Local References

- `./SETUP.md` — як активувати локальне середовище.
- `./REUSABLE_FROM_PREVIOUS_LESSONS.md` — що можна переiспользувати з lesson-06..09.
- `./.env.example` — повний список env vars з коментарями.
- `./pyproject.toml` — версії залежностей (source of truth).
- `./scripts/verify_all.py` — single command для перевірки всіх 7 backend-сервісів.
