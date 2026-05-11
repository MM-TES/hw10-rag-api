"""verify_qdrant.py — confirm QDRANT_URL + QDRANT_API_KEY allow create/delete of a 384-dim collection."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def check() -> tuple[bool, str]:
    """Returns (success, message)."""
    load_dotenv()
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url or "xxx" in url:
        return False, "QDRANT_URL missing or placeholder in .env"
    if not key:
        return False, "QDRANT_API_KEY missing in .env"
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        return False, "qdrant-client not installed (pip install qdrant-client)"
    test_collection = "verify_test"
    try:
        client = QdrantClient(url=url, api_key=key, timeout=15)
        if client.collection_exists(test_collection):
            client.delete_collection(test_collection)
        client.create_collection(
            test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        client.delete_collection(test_collection)
    except Exception as e:
        return False, f"Qdrant error: {type(e).__name__}: {e}"
    return True, "Created/deleted verify_test collection (384-dim cosine)"


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] Qdrant     — {detail}")
    sys.exit(0 if ok else 1)
