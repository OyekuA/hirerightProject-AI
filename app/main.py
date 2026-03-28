import os
import structlog
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from app.logging_config import configure_logging

from app.config import get_settings
from app.middleware.auth import APIKeyMiddleware
from app.middleware.correlation import CorrelationIdMiddleware
from app.clients.gemini import GeminiUnavailableError
from app.routers import (
    ingestion,
    assessment,
    scoring,
    recommend,
    career,
    jd,
)
from app.clients.dependencies import get_qdrant_client, get_gemini_client, get_rate_limiter
from app.services.ingestion_store import IngestionStatusStore
from app.services.callback_client import CallbackClient


logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()

    configure_logging()
    if settings.SENTRY_DSN:
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=1.0,
        )

    lock_acquired = False
    lock_path = "/tmp/hireright.lock"
    if settings.ENFORCE_SINGLE_REPLICA:
        attempt_lock = True
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r") as f:
                    pid_str = f.read().strip()
                if pid_str:
                    pid = int(pid_str)
                    os.kill(pid, 0)
                    logger.warning(
                        "Multi‑instance execution detected – in‑memory state will be inconsistent"
                    )
                    attempt_lock = False
                else:
                    raise ValueError("Empty PID")
            except (OSError, ValueError, FileNotFoundError):
                try:
                    os.unlink(lock_path)
                    logger.debug("Removed stale lock file", lock_path=lock_path)
                except Exception:
                    pass
                attempt_lock = True
        
        if attempt_lock:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                pid = os.getpid()
                os.write(fd, str(pid).encode())
                os.close(fd)
                lock_acquired = True
                logger.info("Single-replica lock acquired", lock_path=lock_path)
            except FileExistsError:
                logger.warning(
                    "Multi‑instance execution detected – in‑memory state will be inconsistent"
                )
            except Exception as e:
                logger.error("Failed to create single‑replica lock", error=str(e))

    qdrant_client = get_qdrant_client()
    qdrant_client.ensure_collections()
    logger.info("Qdrant collections ready")

    gemini_client = get_gemini_client()
    logger.info("Gemini client ready")

    store = IngestionStatusStore(settings.INGEST_STATUS_STORE_PATH)
    callback_client = CallbackClient(
        hmac_secret=settings.CALLBACK_HMAC_SECRET,
        max_retries=settings.CALLBACK_MAX_RETRIES,
        retry_base_seconds=settings.CALLBACK_RETRY_BASE_SECONDS,
    )
    incomplete = store.get_all_incomplete()
    if incomplete:
        logger.info("Found incomplete ingestion records, marking as failed", count=len(incomplete))
        for record in incomplete:
            store.update(
                record.event_id,
                status="failed",
                error_summary="interrupted_by_restart",
            )
            delivered = callback_client.send(
                callback_url=record.callback_url,
                event_id=record.event_id,
                entity_type=record.entity_type,
                entity_id=record.entity_id,
                status="failed",
                error="interrupted_by_restart",
            )
            if not delivered:
                store.update(record.event_id, callback_delivery_failed=True)
    else:
        logger.debug("No incomplete ingestion records found")

    yield

    if lock_acquired:
        try:
            os.unlink(lock_path)
            logger.info("Single-replica lock released", lock_path=lock_path)
        except Exception as e:
            logger.error("Failed to remove lock file", error=str(e))



def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    app = FastAPI(
        lifespan=lifespan,
        dependencies=[Depends(api_key_header)],
        security=[{"APIKeyHeader": []}],
    )

    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    limiter = get_rate_limiter().limiter
    app.state.limiter = limiter

    async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        logger.error("Rate limit exceeded", exc_info=exc)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Router registration (all under /api/ai)
    app.include_router(ingestion.router, prefix="/api/ai")
    app.include_router(assessment.router, prefix="/api/ai")
    app.include_router(scoring.router, prefix="/api/ai")
    app.include_router(recommend.router, prefix="/api/ai")
    app.include_router(career.router, prefix="/api/ai")
    app.include_router(jd.router, prefix="/api/ai")

    @app.get("/health")
    async def health():
        """Health check endpoint (unauthenticated)."""
        return {"status": "ok"}

    @app.exception_handler(GeminiUnavailableError)
    async def gemini_unavailable_handler(request: Request, exc: GeminiUnavailableError):
        logger.error("AI service unavailable", exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "AI service temporarily unavailable"},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch any unhandled exception and return a generic 500."""
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()