"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI

from app.config import settings
from app.deps import lifespan


def _configure_logging() -> None:
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


_configure_logging()

app = FastAPI(
    title="HW10 RAG API",
    version="0.1.0",
    description="Production RAG API for The Twelve-Factor App methodology.",
    lifespan=lifespan,
)


from app.routes.health import router as health_router  # noqa: E402
from app.routes.chat import router as chat_router  # noqa: E402

app.include_router(health_router, tags=["health"])
app.include_router(chat_router, tags=["chat"])
