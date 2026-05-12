"""Budget guard - prevents experiments from spending more than configured cap."""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()


class BudgetExceededError(Exception):
    pass


class BudgetGuard:
    def __init__(self, hard_stop_usd: float = 5.00):
        self.hard_stop = hard_stop_usd
        self._engine = None
        self._baseline: float | None = None

    @property
    def engine(self):
        if not self._engine:
            db_url = os.environ["DATABASE_URL"].replace("?pgbouncer=true", "")
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            self._engine = create_async_engine(db_url, connect_args={"statement_cache_size": 0})
        return self._engine

    async def get_spent_usd(self) -> float:
        async with self.engine.connect() as conn:
            r = await conn.execute(text("SELECT COALESCE(SUM(cost_usd), 0) FROM request_logs"))
            return float(r.scalar() or 0)

    async def set_baseline(self) -> float:
        """Snapshot current DB spend; future check()/remaining() are computed against this."""
        self._baseline = await self.get_spent_usd()
        return self._baseline

    async def spent_since_baseline(self) -> float:
        if self._baseline is None:
            await self.set_baseline()
        return max(0.0, await self.get_spent_usd() - (self._baseline or 0.0))

    async def check(self, projected_usd: float = 0.0) -> None:
        spent = await self.spent_since_baseline()
        if spent + projected_usd >= self.hard_stop:
            raise BudgetExceededError(
                f"Budget hit: spent_since_baseline=${spent:.4f}, "
                f"projected=${projected_usd:.4f}, cap=${self.hard_stop}"
            )

    async def remaining(self) -> float:
        spent = await self.spent_since_baseline()
        return self.hard_stop - spent

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None


guard = BudgetGuard(hard_stop_usd=5.00)


if __name__ == "__main__":
    async def main():
        total_db = await guard.get_spent_usd()
        await guard.set_baseline()
        print(f"Total spend in DB (all-time): ${total_db:.4f}")
        print(f"Phase-2 baseline snapshot:    ${guard._baseline:.4f}")
        print(f"Phase-2 hard cap:             ${guard.hard_stop:.4f}")
        print(f"Phase-2 spent since baseline: ${await guard.spent_since_baseline():.4f}")
        print(f"Phase-2 remaining:            ${await guard.remaining():.4f}")
        await guard.close()
    asyncio.run(main())
