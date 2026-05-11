"""POST /index/rebuild — enterprise-only re-indexation trigger."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.auth import verify_api_key

log = structlog.get_logger()

router = APIRouter()


def _run_index_subprocess() -> None:
    """Run scripts/index.py in a subprocess (sentence-transformers is heavy)."""
    import subprocess
    project_root = Path(__file__).resolve().parent.parent.parent
    cmd = [sys.executable, str(project_root / "scripts" / "index.py")]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        log.info(
            "admin_reindex_done",
            returncode=result.returncode,
            stdout_tail=result.stdout[-500:],
        )
    except Exception as e:
        log.error("admin_reindex_failed", error=str(e))


@router.post("/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_index(
    background: BackgroundTasks,
    key_data: dict = Depends(verify_api_key),
) -> dict:
    if key_data["tier"] != "enterprise":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="enterprise tier required",
        )
    background.add_task(asyncio.to_thread, _run_index_subprocess)
    return {"status": "accepted", "message": "index rebuild scheduled in background"}
