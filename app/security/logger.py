"""Append suspicious request/response events to logs/ files."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
REQUESTS_FILE = LOGS_DIR / "suspicious_requests.log"
RESPONSES_FILE = LOGS_DIR / "suspicious_responses.log"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _append(path: Path, line: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_suspicious_input(api_key: str, pattern: str, message: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{ts}\t{_hash(api_key)}\t{pattern}\t{message[:100].replace(chr(10), ' ')}"
    _append(REQUESTS_FILE, line)


def log_suspicious_output(api_key: str, response: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{ts}\t{_hash(api_key)}\t{response[:200].replace(chr(10), ' ')}"
    _append(RESPONSES_FILE, line)
