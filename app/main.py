import os
import asyncio
import structlog
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from app.logging_config import configure_logging

from app.config import get_settings
from app.middleware.correlation import CorrelationIdMiddleware
from app.clients.llm import LLMUnavailableError
from app.auth import verify_api_key
from app.routers import (
    ingestion,
    assessment,
    scoring,
    recommend,
    career,
    jd,
)
from app.clients.dependencies import (
    get_qdrant_client,
    get_llm_client,
    get_rate_limiter,
    get_ingestion_store,
    get_callback_client,
    get_ingest_queue,
)
from app.services.ingestion_service import run_candidate_ingestion, run_job_ingestion


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
    lock_path = "/data/ingest_status/.hireright.lock"
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

    llm_client = get_llm_client()
    logger.info("LLM client ready")

    store = get_ingestion_store()
    callback_client = get_callback_client()
    incomplete = store.get_all_incomplete()
    if incomplete:
        logger.info("Found incomplete ingestion records, marking as failed", count=len(incomplete))
        RECOVERY_BUDGET = 30.0  # seconds
        start_time = asyncio.get_event_loop().time()
        deferred = []
        for record in incomplete:
            store.update(
                record.event_id,
                status="failed",
                error_summary="interrupted_by_restart",
            )
            if asyncio.get_event_loop().time() - start_time >= RECOVERY_BUDGET:
                deferred.append(record)
                logger.debug("Recovery budget exceeded, deferring callback", event_id=record.event_id)
                continue
            try:
                delivered = await asyncio.wait_for(
                    callback_client.send(
                        callback_url=record.callback_url,
                        event_id=record.event_id,
                        entity_type=record.entity_type,
                        entity_id=record.entity_id,
                        status="failed",
                        error="interrupted_by_restart",
                    ),
                    timeout=callback_client.timeout_seconds,
                )
                if not delivered:
                    store.update(record.event_id, callback_delivery_failed=True)
            except asyncio.TimeoutError:
                logger.warning("Callback timeout during startup recovery", event_id=record.event_id)
                deferred.append(record)
            except Exception as e:
                logger.error("Unexpected error during startup callback", event_id=record.event_id, error=str(e))
                deferred.append(record)
        if deferred:
            logger.info("Callbacks deferred due to recovery budget", count=len(deferred), event_ids=[r.event_id for r in deferred])
            async def process_deferred_callbacks(records):
                sem = asyncio.Semaphore(5)
                async def process_one(rec):
                    async with sem:
                        try:
                            delivered = await callback_client.send(
                                callback_url=rec.callback_url,
                                event_id=rec.event_id,
                                entity_type=rec.entity_type,
                                entity_id=rec.entity_id,
                                status="failed",
                                error="interrupted_by_restart",
                            )
                            if not delivered:
                                store.update(rec.event_id, callback_delivery_failed=True)
                        except Exception as e:
                            logger.error("Deferred callback failed", event_id=rec.event_id, error=str(e))
                await asyncio.gather(*(process_one(r) for r in records))
            asyncio.create_task(process_deferred_callbacks(deferred))
    else:
        logger.debug("No incomplete ingestion records found")

    # ── Queue worker ──────────────────────────────────────────────
    ingest_queue = get_ingest_queue()

    async def process_queue_entry(
        entry,
        ingest_queue,
        qdrant_client,
        llm_client,
        store,
        callback_client,
        settings,
    ):
        try:
            payload = entry.payload
            if entry.entity_type == "candidate":
                await run_candidate_ingestion(
                    candidate_id=entry.entity_id,
                    cv_url=payload["cv_url"],
                    profile_data=payload["profile_data"],
                    callback_url=entry.callback_url,
                    event_id=entry.event_id,
                    qdrant=qdrant_client,
                    llm=llm_client,
                    store=store,
                    callback_client=callback_client,
                    ingest_queue=None,
                    suppress_callback=True,
                )
            elif entry.entity_type == "job":
                await run_job_ingestion(
                    job_id=entry.entity_id,
                    jd_text=payload["jd_text"],
                    metadata=payload["metadata"],
                    callback_url=entry.callback_url,
                    event_id=entry.event_id,
                    qdrant=qdrant_client,
                    llm=llm_client,
                    store=store,
                    callback_client=callback_client,
                    ingest_queue=None,
                    suppress_callback=True,
                )

            record = store.get_by_event_id(entry.event_id)
            if record is not None and record.status == "success":
                ingest_queue.remove(entry.event_id)
                # Send success callback now that the queue replay succeeded
                await callback_client.send(
                    callback_url=entry.callback_url,
                    event_id=entry.event_id,
                    entity_type=entry.entity_type,
                    entity_id=entry.entity_id,
                    status="success",
                    error=None,
                )
                logger.info("Queue entry processed successfully", event_id=entry.event_id)
            else:
                requeued = ingest_queue.requeue(
                    entry,
                    settings.INGEST_QUEUE_MAX_RETRIES,
                    settings.INGEST_QUEUE_BACKOFF_BASE_SECONDS,
                )
                if not requeued:
                    # Entry was moved to dead letter — send final failure callback
                    await callback_client.send(
                        callback_url=entry.callback_url,
                        event_id=entry.event_id,
                        entity_type=entry.entity_type,
                        entity_id=entry.entity_id,
                        status="failed",
                        error="max_queue_retries_exceeded",
                    )
        except Exception as e:
            logger.error("Queue entry processing error", event_id=entry.event_id, error=str(e))

    async def queue_worker(ingest_queue, qdrant_client, llm_client, store, callback_client, settings):
        while True:
            await asyncio.sleep(settings.INGEST_QUEUE_POLL_INTERVAL_SECONDS)
            try:
                entries = ingest_queue.get_due_entries()
                for entry in entries:
                    asyncio.create_task(
                        process_queue_entry(
                            entry, ingest_queue, qdrant_client, llm_client, store, callback_client, settings
                        )
                    )
            except Exception as e:
                logger.error("Queue worker poll error", error=str(e))

    _worker_task = asyncio.create_task(
        queue_worker(ingest_queue, qdrant_client, llm_client, store, callback_client, settings)
    )
    logger.info("Ingest queue worker started", poll_interval=settings.INGEST_QUEUE_POLL_INTERVAL_SECONDS)

    yield

    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        logger.info("Ingest queue worker stopped")

    if lock_acquired:
        try:
            os.unlink(lock_path)
            logger.info("Single-replica lock released", lock_path=lock_path)
        except Exception as e:
            logger.error("Failed to remove lock file", error=str(e))



def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="HireRight AI API", 
        description="Core AI microservice for candidate ingestion, scoring, and career recommendations.",
        version="1.0.0",
        
        lifespan=lifespan,
        openapi_url="/openapi.json", 
        docs_url="/", 
        redoc_url=None,
    )

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

    app.include_router(ingestion.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])
    app.include_router(assessment.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])
    app.include_router(scoring.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])
    app.include_router(recommend.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])
    app.include_router(career.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])
    app.include_router(jd.router, prefix="/api/ai", dependencies=[Depends(verify_api_key)])

    @app.get("/health")
    async def health():
        """Health check endpoint (unauthenticated)."""
        return {"status": "ok"}

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
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