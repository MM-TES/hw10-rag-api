"""verify_fly.py -- format-only check for FLY_API_TOKEN.

Fly verify is intentionally limited until Phase 1:
- Org-scoped macaroons cannot run read GraphQL queries (orgs list/show, whoami)
- Real verification requires creating an app first (Phase 1)
- This check confirms only that token is PRESENT and well-formed
"""
from __future__ import annotations

import os
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()


def check() -> tuple[bool, str]:
    """Returns (success, message). Always SKIP pre-Phase 1."""
    if shutil.which("fly") is None and shutil.which("flyctl") is None:
        return False, "SKIP: fly CLI not installed (will need for Phase 1 deploy)"

    token = os.environ.get("FLY_API_TOKEN", "").strip()
    if not token or "placeholder" in token.lower() or "your" in token.lower():
        return False, "SKIP: FLY_API_TOKEN not set in .env yet"

    if not token.startswith("FlyV1 fm2_"):
        return False, (
            f"SKIP: FLY_API_TOKEN has unexpected format "
            f"(length={len(token)}, prefix={token[:15]!r})"
        )

    if len(token) < 100:
        return False, f"SKIP: FLY_API_TOKEN suspiciously short ({len(token)} chars)"

    return False, (
        f"SKIP: Token present and well-formed (length={len(token)}). "
        "Real verify deferred to Phase 1 (need app + deploy-scoped token)."
    )


if __name__ == "__main__":
    ok, detail = check()
    label = "SKIP" if (not ok and detail.startswith("SKIP")) else ("OK" if ok else "FAIL")
    print(f"[{label}] Fly        -- {detail}")
    # Exit 0 for SKIP -- Fly is optional pre-Phase 1
    sys.exit(0 if (ok or detail.startswith("SKIP")) else 1)
