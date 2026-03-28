"""Singleton accessors for infrastructure clients.

This module provides FastAPI‑compatible dependency functions that return
shared instances of QdrantClient, GeminiClient, CacheBackend, and RateLimiterBackend.
Each singleton is lazily initialized on first call using configuration from
`get_settings()`.
"""

from typing import Optional

from fastapi import Depends

from app.config import get_settings
from app.clients.qdrant import QdrantClient, CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient
from app.clients.cache import CacheBackend, TTLCacheBackend
from app.clients.rate_limiter import RateLimiterBackend, SlowAPIRateLimiterBackend
from app.services.callback_client import CallbackClient
from app.services.ingestion_store import IngestionStatusStore

_qdrant_instance: Optional[QdrantClient] = None
_gemini_instance: Optional[GeminiClient] = None
_cache_instance: Optional[CacheBackend] = None
_rate_limiter_instance: Optional[RateLimiterBackend] = None
_callback_client_instance: Optional[CallbackClient] = None
_ingestion_store_instance: Optional[IngestionStatusStore] = None


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


def get_gemini_client() -> GeminiClient:
    """Return a singleton GeminiClient configured from environment variables."""
    global _gemini_instance
    if _gemini_instance is None:
        settings = get_settings()
        _gemini_instance = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            generation_cooldown=settings.GENERATION_BREAKER_COOLDOWN_SECONDS,
            embedding_cooldown=settings.EMBEDDING_BREAKER_COOLDOWN_SECONDS,
        )
    return _gemini_instance


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
            max_retries=settings.CALLBACK_MAX_RETRIES,
            retry_base_seconds=settings.CALLBACK_RETRY_BASE_SECONDS,
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


__all__ = [
    "get_qdrant_client",
    "get_gemini_client",
    "get_cache_backend",
    "get_rate_limiter",
    "get_callback_client",
    "get_ingestion_store",
    "CANDIDATES_COLLECTION",
    "JOBS_COLLECTION",
]