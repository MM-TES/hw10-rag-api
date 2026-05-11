"""verify_all.py — run all per-service checks and print a consolidated pass/fail table.

Imports `check()` functions from each verify_<service>.py module so logic is not duplicated.

Exit code:
    0 — all required services are green (Fly may be SKIP).
    1 — any required service failed.

Usage:
    python scripts/verify_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root: `python scripts/verify_all.py`
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Each entry: (label, module_attr_path, required)
SERVICES = [
    ("OpenRouter", "verify_openrouter:check", True),
    ("Anthropic",  "verify_anthropic:check",  True),
    ("Qdrant",     "verify_qdrant:check",     True),
    ("Redis",      "verify_redis:check",      True),
    ("Postgres",   "verify_postgres:check",   True),
    ("Langfuse",   "verify_langfuse:check",   True),
    ("Fly",        "verify_fly:check",        False),  # optional until deployment
]


def _run_one(module_attr: str) -> tuple[bool, str]:
    mod_name, attr = module_attr.split(":")
    try:
        mod = __import__(mod_name)
        check_fn = getattr(mod, attr)
    except Exception as e:
        return False, f"could not import {module_attr}: {e}"
    try:
        return check_fn()
    except Exception as e:
        return False, f"check raised: {type(e).__name__}: {e}"


def main() -> int:
    print(f"{'Service':<12}  {'Status':<6}  Detail")
    print("-" * 70)
    passed = 0
    failed = 0
    skipped = 0
    for label, module_attr, required in SERVICES:
        ok, detail = _run_one(module_attr)
        if ok:
            status = "OK"
            passed += 1
        elif not required and detail.startswith("SKIP"):
            status = "SKIP"
            skipped += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"{label:<12}  [{status:<4}]  {detail}")
    print("-" * 70)
    total_required = sum(1 for _, _, req in SERVICES if req)
    summary = f"{passed}/{total_required} required passed"
    if skipped:
        summary += f" ({skipped} optional skipped)"
    if failed == 0:
        print(f"{summary}. Setup is READY.")
        return 0
    print(f"{summary}. {failed} FAILED. Setup is NOT ready -- fix the failing services and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
