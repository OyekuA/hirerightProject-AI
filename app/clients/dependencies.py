import threading
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
import structlog

from app.config import get_settings
from app.clients.qdrant import QdrantClient, CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient
from app.clients.cache import CacheBackend, TTLCacheBackend
from app.clients.rate_limiter import RateLimiterBackend, SlowAPIRateLimiterBackend
from app.clients.meeting_bot import MeetingBotClient, RecallAIClient
from app.services.callback_client import CallbackClient
from app.services.ingestion_store import IngestionStatusStore
from app.services.ingest_queue import IngestQueue
from app.services.screening_store import BatchScreeningStore
from app.services.interview_session_store import InterviewSessionStore

logger = structlog.get_logger()

_qdrant_instance: Optional[QdrantClient] = None
_qdrant_lock = threading.Lock()
_llm_instance: Optional[LLMClient] = None
_llm_lock = threading.Lock()
_cache_instance: Optional[CacheBackend] = None
_cache_lock = threading.Lock()
_rate_limiter_instance: Optional[RateLimiterBackend] = None
_rate_limiter_lock = threading.Lock()
_callback_client_instance: Optional[CallbackClient] = None
_callback_lock = threading.Lock()
_ingestion_store_instance: Optional[IngestionStatusStore] = None
_store_lock = threading.Lock()
_ingest_queue_instance: Optional[IngestQueue] = None
_queue_lock = threading.Lock()
_screening_store_instance: Optional[BatchScreeningStore] = None
_screening_store_lock = threading.Lock()
_meeting_bot_client_instance: Optional[MeetingBotClient] = None
_meeting_bot_client_lock = threading.Lock()
_interview_session_store_instance: Optional[InterviewSessionStore] = None
_interview_session_store_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    global _qdrant_instance
    if _qdrant_instance is None:
        with _qdrant_lock:
            if _qdrant_instance is None:
                settings = get_settings()
                _qdrant_instance = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
    return _qdrant_instance


def get_llm_client() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
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
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                settings = get_settings()
                _cache_instance = TTLCacheBackend(
                    maxsize=1000,
                    ttl=settings.CACHE_TTL_SECONDS,
                )
    return _cache_instance


def get_rate_limiter() -> RateLimiterBackend:
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        with _rate_limiter_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = SlowAPIRateLimiterBackend(default_limits=None)
    return _rate_limiter_instance


def get_callback_client() -> CallbackClient:
    global _callback_client_instance
    if _callback_client_instance is None:
        with _callback_lock:
            if _callback_client_instance is None:
                settings = get_settings()
                _callback_client_instance = CallbackClient(
                    hmac_secret=settings.CALLBACK_HMAC_SECRET,
                    max_attempts=settings.CALLBACK_MAX_ATTEMPTS,
                    retry_base_seconds=settings.CALLBACK_RETRY_BASE_SECONDS,
                    timeout_seconds=settings.CALLBACK_TIMEOUT_SECONDS,
                    signature_ttl_seconds=settings.CALLBACK_SIGNATURE_TTL_SECONDS,
                )
    return _callback_client_instance


def get_ingestion_store() -> IngestionStatusStore:
    global _ingestion_store_instance
    if _ingestion_store_instance is None:
        with _store_lock:
            if _ingestion_store_instance is None:
                settings = get_settings()
                _ingestion_store_instance = IngestionStatusStore(
                    store_path=settings.INGEST_STATUS_STORE_PATH,
                )
    return _ingestion_store_instance


def get_ingest_queue() -> IngestQueue:
    global _ingest_queue_instance
    if _ingest_queue_instance is None:
        with _queue_lock:
            if _ingest_queue_instance is None:
                settings = get_settings()
                base = Path(settings.INGEST_STATUS_STORE_PATH)
                _ingest_queue_instance = IngestQueue(
                    queue_path=str(base / "failed_queue"),
                    dead_letter_path=str(base / "dead_letter"),
                )
    return _ingest_queue_instance


def get_screening_store() -> BatchScreeningStore:
    global _screening_store_instance
    if _screening_store_instance is None:
        with _screening_store_lock:
            if _screening_store_instance is None:
                settings = get_settings()
                base = Path(settings.INGEST_STATUS_STORE_PATH)
                _screening_store_instance = BatchScreeningStore(
                    store_path=str(base / "screening_batches"),
                )
    return _screening_store_instance


def get_meeting_bot_client() -> MeetingBotClient:
    global _meeting_bot_client_instance
    if _meeting_bot_client_instance is None:
        with _meeting_bot_client_lock:
            if _meeting_bot_client_instance is None:
                settings = get_settings()
                try:
                    _meeting_bot_client_instance = RecallAIClient(
                        api_key=settings.RECALL_AI_API_KEY,
                        region=settings.RECALL_AI_REGION,
                    )
                except ValueError as exc:
                    logger.error(
                        "Invalid Recall.ai configuration",
                        region=settings.RECALL_AI_REGION,
                        error=str(exc),
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid Recall.ai region configuration: {exc}",
                    )
    return _meeting_bot_client_instance


def get_interview_session_store() -> InterviewSessionStore:
    global _interview_session_store_instance
    if _interview_session_store_instance is None:
        with _interview_session_store_lock:
            if _interview_session_store_instance is None:
                settings = get_settings()
                _interview_session_store_instance = InterviewSessionStore(
                    store_path=settings.INTERVIEW_SESSION_STORE_PATH,
                )
    return _interview_session_store_instance


__all__ = [
    "get_qdrant_client",
    "get_llm_client",
    "get_cache_backend",
    "get_rate_limiter",
    "get_callback_client",
    "get_ingestion_store",
    "get_ingest_queue",
    "get_screening_store",
    "get_meeting_bot_client",
    "get_interview_session_store",
    "CANDIDATES_COLLECTION",
    "JOBS_COLLECTION",
]
