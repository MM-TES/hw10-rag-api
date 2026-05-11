# SETUP.md — Activate the local environment

This is the **launcher**. The detailed walkthrough (registration on 7 cloud services, getting API keys, copying values into `.env`) lives in `../_tasks/SETUP_CHECKLIST.md` — do that first, then come back here.

---

## 1. Prerequisites

- Python 3.11+
- Conda (Miniconda or Anaconda)
- Git
- PowerShell (Windows). Bash works too via `Bash` tool / WSL.

## 2. Create and activate the environment

```powershell
conda create -n hw10 python=3.11 -y
conda activate hw10
```

## 3. Install dependencies

Editable install with dev extras (preferred — exposes `app` package and dev tooling):

```powershell
pip install -e ".[dev]"
```

Or, alternatively, plain requirements file:

```powershell
pip install -r requirements.txt
```

For Phase 2 (report generation), additionally:

```powershell
pip install -e ".[report]"
```

## 4. Create `.env`

```powershell
Copy-Item .env.example .env
```

Then **fill in real values per `../_tasks/SETUP_CHECKLIST.md`** — that document walks through the 7 service registrations (OpenRouter, Anthropic, Qdrant Cloud, Upstash Redis, Supabase Postgres, Langfuse, Fly.io) and how to extract each credential.

## 5. Verify all backends

```powershell
python scripts\verify_all.py
```

Expected output:

```
Service       Status    Detail
-----------   --------  ---------------------------------
OpenRouter    [OK]      ...
Anthropic     [OK]      ...
Qdrant        [OK]      ...
Redis         [OK]      ...
Postgres      [OK]      ...
Langfuse      [OK]      ...
Fly           [OK]      ... (or [SKIP] if you postpone Fly to Phase 2)
6/7 passed (1 skipped). Setup is ready.
```

You can also run a single per-service check while you fill in `.env`:

```powershell
python scripts\verify_qdrant.py        # check just Qdrant
python scripts\verify_postgres.py      # check just Postgres
```

## 6. Setup is complete when

- ✅ `python scripts\verify_all.py` shows 6/7 or 7/7 green (Fly is optional for Phase 1).
- ✅ `git status` shows no `.env` (it's `.gitignore`'d).
- ✅ You can read `CLAUDE.md` end-to-end and it makes sense.

Now you can launch Phase 1 (`../_tasks/CLAUDE_CODE_PHASE1.md`).

---

## Troubleshooting (4 most common pitfalls)

For full troubleshooting see `../_tasks/SETUP_CHECKLIST.md` and `CLAUDE.md §When Things Break`.

1. **Postgres connection error / asyncpg / IPv6:**
   `DATABASE_URL` must use Supabase's **pooler** (port `6543`, not `5432`) AND start with `postgresql+asyncpg://`. Direct connections kill Fly.io over IPv6. Special chars in password must be URL-encoded (`@` → `%40`).

2. **Upstash returns 400 on Lua script:** Upstash REST does **not** support Lua. Use `INCRBY` + `EXPIRE` separately (the rate-limit module already does this).

3. **Qdrant has no `expire_at` / TTL:** Qdrant has no built-in TTL. The semantic cache writes a `expire_at` (Unix ts) field into payload and filters by it on read.

4. **`sentence-transformers` SSL error on first download (Windows):** common Windows issue. Workaround: pre-download the model via plain `pip download` or set `HF_HUB_DISABLE_SSL_VERIFY=1` temporarily (then unset). Don't fight this in Claude Code — handle it manually.
