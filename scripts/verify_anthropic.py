"""verify_anthropic.py — confirm ANTHROPIC_API_KEY works against Claude Haiku."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def check() -> tuple[bool, str]:
    """Returns (success, message)."""
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or key.startswith("sk-ant-..."):
        return False, "ANTHROPIC_API_KEY missing or placeholder in .env"
    try:
        import anthropic  # lazy import — keeps this module importable without the package
    except ImportError:
        return False, "anthropic package not installed (pip install anthropic)"
    try:
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say OK"}],
        )
    except Exception as e:
        return False, f"API error: {type(e).__name__}: {e}"
    text = r.content[0].text if r.content else ""
    return True, f"Returned: {text!r}"


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] Anthropic  — {detail}")
    sys.exit(0 if ok else 1)
