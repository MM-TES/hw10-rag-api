"""In-memory circuit breaker per model (failures within window + open duration)."""
from __future__ import annotations

import time
from collections import defaultdict

from app.config import settings


class CircuitBreaker:
    def __init__(
        self,
        threshold: int | None = None,
        window_s: int | None = None,
        open_s: int | None = None,
    ):
        self.threshold = threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.window_s = window_s or settings.CIRCUIT_BREAKER_WINDOW_SECONDS
        self.open_s = open_s or settings.CIRCUIT_BREAKER_OPEN_SECONDS
        self._failures: dict[str, list[float]] = defaultdict(list)
        self._open_until: dict[str, float] = {}

    def is_open(self, model: str) -> bool:
        now = time.time()
        until = self._open_until.get(model)
        if until and now < until:
            return True
        if until and now >= until:
            self._open_until.pop(model, None)
            self._failures[model].clear()
        return False

    def record_failure(self, model: str) -> None:
        now = time.time()
        bucket = self._failures[model]
        bucket.append(now)
        cutoff = now - self.window_s
        self._failures[model] = [t for t in bucket if t >= cutoff]
        if len(self._failures[model]) >= self.threshold:
            self._open_until[model] = now + self.open_s

    def reset(self, model: str) -> None:
        self._failures.pop(model, None)
        self._open_until.pop(model, None)


breaker = CircuitBreaker()
