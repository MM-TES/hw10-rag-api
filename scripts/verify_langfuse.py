"""verify_langfuse.py — confirm Langfuse SDK v4 can create + flush a span.

v4 (March 2026) removed the v2 trace()/span() helpers. The supported entrypoint
is get_client() + start_as_current_observation(as_type="span", ...).
Docs: https://langfuse.com/docs/observability/sdk/overview
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def check() -> tuple[bool, str]:
    """Returns (success, message)."""
    load_dotenv()
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    if not pk or pk.startswith("pk-lf-..."):
        return False, "LANGFUSE_PUBLIC_KEY missing or placeholder in .env"
    if not sk or sk.startswith("sk-lf-..."):
        return False, "LANGFUSE_SECRET_KEY missing or placeholder in .env"
    if not host:
        return False, "LANGFUSE_HOST (or LANGFUSE_BASE_URL) missing in .env"
    try:
        from langfuse import get_client
    except ImportError:
        return False, "langfuse package not installed"
    try:
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="span",
            name="verify_test",
            input={"query": "ping"},
        ) as span:
            span.update(output={"status": "ok"})
        langfuse.flush()
    except Exception as e:
        return False, f"Langfuse v4 API error: {type(e).__name__}: {str(e)[:200]}"
    return True, f"Span 'verify_test' created and flushed to {host}"


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] Langfuse   — {detail}")
    sys.exit(0 if ok else 1)
