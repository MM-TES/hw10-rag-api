"""verify_redis.py — confirm Upstash Redis REST API responds to /set + /get."""
from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv


def check() -> tuple[bool, str]:
    """Returns (success, message)."""
    load_dotenv()
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not url or "xxx" in url:
        return False, "UPSTASH_REDIS_REST_URL missing or placeholder in .env"
    if not token:
        return False, "UPSTASH_REDIS_REST_TOKEN missing in .env"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        # SET test_key test_value
        r1 = httpx.post(f"{url}/set/verify_key/verify_value", headers=headers, timeout=10.0)
        if r1.status_code != 200 or r1.json().get("result") != "OK":
            return False, f"SET failed: {r1.status_code} {r1.text[:200]}"
        # GET test_key
        r2 = httpx.get(f"{url}/get/verify_key", headers=headers, timeout=10.0)
        if r2.status_code != 200:
            return False, f"GET failed: {r2.status_code} {r2.text[:200]}"
        if r2.json().get("result") != "verify_value":
            return False, f"GET returned unexpected: {r2.json()}"
        # cleanup
        httpx.post(f"{url}/del/verify_key", headers=headers, timeout=10.0)
    except httpx.HTTPError as e:
        return False, f"HTTP error: {e}"
    return True, "SET/GET round-trip OK"


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] Redis      — {detail}")
    sys.exit(0 if ok else 1)
