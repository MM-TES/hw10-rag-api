"""Pydantic Settings — env-driven config singleton."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    OPENROUTER_API_KEY: str
    ANTHROPIC_API_KEY: str = ""

    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_CHUNKS_COLLECTION: str = "chunks_collection"
    QDRANT_CACHE_COLLECTION: str = "cache_collection"

    UPSTASH_REDIS_REST_URL: str
    UPSTASH_REDIS_REST_TOKEN: str

    DATABASE_URL: str

    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    FLY_API_TOKEN: str = ""

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CONCURRENT_LLM_CALLS: int = 20
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_TTL_SECONDS: int = 3600
    LLM_TIMEOUT_SECONDS: int = 15
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_WINDOW_SECONDS: int = 60
    CIRCUIT_BREAKER_OPEN_SECONDS: int = 60
    ENABLE_CACHE: bool = True
    ENABLE_FALLBACK: bool = True
    ENABLE_INJECTION_DEFENSE: bool = True

    EXPERIMENT_BUDGET_USD: float = 10.0
    EXPERIMENT_HARD_STOP_USD: float = 9.5
    EXPERIMENT_PER_REQUEST_TIMEOUT_S: int = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
