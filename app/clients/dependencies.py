"""Singleton accessors for infrastructure clients.

This module provides FastAPI‑compatible dependency functions that return
shared instances of QdrantClient, LLMClient, CacheBackend, and RateLimiterBackend.
Each singleton is lazily initialized on first call using configuration from
`get_settings()`.
"""

from pathlib import Path
from typing import Optional

from fastapi import Depends

from app.config import get_settings
from app.clients.qdrant import QdrantClient, CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient
from app.clients.cache import CacheBackend, TTLCacheBackend
from app.clients.rate_limiter import RateLimiterBackend, SlowAPIRateLimiterBackend
from app.services.callback_client import CallbackClient
from app.services.ingestion_store import IngestionStatusStore
from app.services.ingest_queue import IngestQueue

_qdrant_instance: Optional[QdrantClient] = None
_llm_instance: Optional[LLMClient] = None
_cache_instance: Optional[CacheBackend] = None
_rate_limiter_instance: Optional[RateLimiterBackend] = None
_callback_client_instance: Optional[CallbackClient] = None
_ingestion_store_instance: Optional[IngestionStatusStore] = None
_ingest_queue_instance: Optional[IngestQueue] = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient configured from environment variables."""
    global _qdrant_instance
    if _qdrant_instance is None:
        settings = get_settings()
        _qdrant_instance = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
    return _qdrant_instance


def get_llm_client() -> LLMClient:
    """Return a singleton LLMClient configured from environment variables."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = LLMClient(
            model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
            generation_cooldown=settings.GENERATION_BREAKER_COOLDOWN_SECONDS,
            embedding_cooldown=settings.EMBEDDING_BREAKER_COOLDOWN_SECONDS,
            generation_timeout=settings.LLM_GENERATION_TIMEOUT_SECONDS,
            embedding_timeout=settings.LLM_EMBEDDING_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            retry_backoff_base=settings.LLM_RETRY_BACKOFF_BASE_SECONDS,
            api_key=settings.LLM_API_KEY,
        )
    return _llm_instance


def get_cache_backend() -> CacheBackend:
    """Return a singleton CacheBackend (TTLCache‑based) with default TTL."""
    global _cache_instance
    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = TTLCacheBackend(
            maxsize=1000,
            ttl=settings.CACHE_TTL_SECONDS,
        )
    return _cache_instance


def get_rate_limiter() -> RateLimiterBackend:
    """Return a singleton rate‑limiter backend with no default limits.

    The returned instance's `.limiter` property can be used to decorate routes.
    """
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        _rate_limiter_instance = SlowAPIRateLimiterBackend(default_limits=None)
    return _rate_limiter_instance


def get_callback_client() -> CallbackClient:
    """Return a singleton CallbackClient configured from environment variables."""
    global _callback_client_instance
    if _callback_client_instance is None:
        settings = get_settings()
        _callback_client_instance = CallbackClient(
            hmac_secret=settings.CALLBACK_HMAC_SECRET,
            max_attempts=settings.CALLBACK_MAX_ATTEMPTS,
            retry_base_seconds=settings.CALLBACK_RETRY_BASE_SECONDS,
            timeout_seconds=settings.CALLBACK_TIMEOUT_SECONDS,
        )
    return _callback_client_instance


def get_ingestion_store() -> IngestionStatusStore:
    """Return a singleton IngestionStatusStore configured from environment variables."""
    global _ingestion_store_instance
    if _ingestion_store_instance is None:
        settings = get_settings()
        _ingestion_store_instance = IngestionStatusStore(
            store_path=settings.INGEST_STATUS_STORE_PATH,
        )
    return _ingestion_store_instance


def get_ingest_queue() -> IngestQueue:
    """Return a singleton IngestQueue configured from environment variables."""
    global _ingest_queue_instance
    if _ingest_queue_instance is None:
        settings = get_settings()
        base = Path(settings.INGEST_STATUS_STORE_PATH)
        _ingest_queue_instance = IngestQueue(
            queue_path=str(base / "failed_queue"),
            dead_letter_path=str(base / "dead_letter"),
        )
    return _ingest_queue_instance


__all__ = [
    "get_qdrant_client",
    "get_llm_client",
    "get_cache_backend",
    "get_rate_limiter",
    "get_callback_client",
    "get_ingestion_store",
    "get_ingest_queue",
    "CANDIDATES_COLLECTION",
    "JOBS_COLLECTION",
]