"""verify_postgres.py — confirm DATABASE_URL allows async connect + write/read/drop."""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv


async def _async_check() -> tuple[bool, str]:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url or "PASSWORD" in url:
        return False, "DATABASE_URL missing or placeholder in .env"
    if "+asyncpg" not in url:
        return False, "DATABASE_URL must use postgresql+asyncpg:// driver (not postgresql://)"
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return False, "sqlalchemy[asyncio] not installed"
    engine = create_async_engine(url, echo=False)
    try:
        async with engine.connect() as conn:
            r = await conn.execute(text("SELECT version()"))
            version = str(r.scalar() or "")[:60]
            await conn.execute(
                text("CREATE TABLE IF NOT EXISTS verify_test (id INT, ts TIMESTAMPTZ DEFAULT now())")
            )
            await conn.execute(text("INSERT INTO verify_test (id) VALUES (1)"))
            await conn.commit()
            r = await conn.execute(text("SELECT COUNT(*) FROM verify_test"))
            rows = r.scalar()
            await conn.execute(text("DROP TABLE verify_test"))
            await conn.commit()
    except Exception as e:
        return False, f"Postgres error: {type(e).__name__}: {e}"
    finally:
        await engine.dispose()
    return True, f"{version} (write/read/drop OK, rows seen: {rows})"


def check() -> tuple[bool, str]:
    return asyncio.run(_async_check())


if __name__ == "__main__":
    ok, detail = check()
    print(f"[{'OK' if ok else 'FAIL'}] Postgres   — {detail}")
    sys.exit(0 if ok else 1)
