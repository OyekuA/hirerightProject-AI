import base64
import os
import asyncio
import secrets
import structlog
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, Response
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
from app.services.ingest_queue import IngestQueue


logger = structlog.get_logger()


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
                correlation_id=None,
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
                correlation_id=None,
            )

        record = store.get_by_event_id(entry.event_id)
        if record is not None and record.status == "success":
            ingest_queue.remove(entry.event_id)
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


async def dead_letter_watcher(queue: IngestQueue) -> None:
    _last_dead_letter_count: int = 0
    _seen_event_ids: set[str] = set()
    settings = get_settings()
    poll_interval = settings.DEAD_LETTER_POLL_INTERVAL_SECONDS
    while True:
        await asyncio.sleep(poll_interval)
        try:
            entries = queue.get_dead_letter_entries()
            new_entries = [e for e in entries if e.event_id not in _seen_event_ids]
            for entry in new_entries:
                _seen_event_ids.add(entry.event_id)
                logger.error(
                    "Ingestion permanently failed — dead letter alert",
                    event_id=entry.event_id,
                    entity_type=entry.entity_type,
                    entity_id=entry.entity_id,
                    queue_retry_count=entry.queue_retry_count,
                    enqueued_at=entry.enqueued_at,
                )
                if sentry_sdk.is_initialized():
                    with sentry_sdk.new_scope() as scope:
                        scope.set_extra("event_id", entry.event_id)
                        scope.set_extra("entity_type", entry.entity_type)
                        scope.set_extra("entity_id", entry.entity_id)
                        scope.set_extra("queue_retry_count", entry.queue_retry_count)
                        scope.set_extra("enqueued_at", entry.enqueued_at)
                        scope.set_level("error")
                        sentry_sdk.capture_message(
                            f"Ingestion permanently failed: {entry.entity_type} {entry.entity_id}"
                        )
            count = len(entries)
            if count == 0 and _last_dead_letter_count > 0:
                logger.info("Dead letter directory cleared", previous_count=_last_dead_letter_count)
                _seen_event_ids.clear()
            _last_dead_letter_count = count
        except Exception as e:
            logger.error("Dead letter watcher error", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    configure_logging()
    if settings.SENTRY_DSN:
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=1.0,
        )

    if settings.ENABLE_DOCS and (not settings.DOCS_USERNAME or not settings.DOCS_PASSWORD):
        logger.warning("ENABLE_DOCS=true but DOCS_USERNAME/DOCS_PASSWORD not set — Swagger UI is publicly accessible")

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
        RECOVERY_BUDGET = 30.0
        start_time = asyncio.get_running_loop().time()
        deferred = []
        for record in incomplete:
            store.update(
                record.event_id,
                status="failed",
                error_summary="interrupted_by_restart",
            )
            if asyncio.get_running_loop().time() - start_time >= RECOVERY_BUDGET:
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

    ingest_queue = get_ingest_queue()

    _worker_task = asyncio.create_task(
        queue_worker(ingest_queue, qdrant_client, llm_client, store, callback_client, settings)
    )
    logger.info("Ingest queue worker started", poll_interval=settings.INGEST_QUEUE_POLL_INTERVAL_SECONDS)

    _dead_letter_task = asyncio.create_task(dead_letter_watcher(ingest_queue))
    logger.info("Dead letter watcher started", poll_interval_seconds=settings.DEAD_LETTER_POLL_INTERVAL_SECONDS)

    yield

    _worker_task.cancel()
    _dead_letter_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        logger.info("Ingest queue worker stopped")
    try:
        await _dead_letter_task
    except asyncio.CancelledError:
        logger.info("Dead letter watcher stopped")

    if lock_acquired:
        try:
            os.unlink(lock_path)
            logger.info("Single-replica lock released", lock_path=lock_path)
        except Exception as e:
            logger.error("Failed to remove lock file", error=str(e))



def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="HireRight AI API",
        description="Core AI microservice for candidate ingestion, scoring, and career recommendations.",
        version="1.0.0",
        
        lifespan=lifespan,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        docs_url="/" if settings.ENABLE_DOCS else None,
        redoc_url=None,
    )

    if settings.ENABLE_DOCS and settings.DOCS_USERNAME and settings.DOCS_PASSWORD:
        @app.middleware("http")
        async def docs_basic_auth(request: Request, call_next):
            protected_paths = {"/openapi.json", "/", "/docs"}
            if request.url.path in protected_paths:
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Basic "):
                    return Response(
                        status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="HireRight AI Docs"'},
                        content="Unauthorized",
                    )
                try:
                    decoded = base64.b64decode(auth[6:]).decode("utf-8")
                    username, _, password = decoded.partition(":")
                except Exception:
                    return Response(
                        status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="HireRight AI Docs"'},
                        content="Unauthorized",
                    )
                valid_user = secrets.compare_digest(username, settings.DOCS_USERNAME)
                valid_pass = secrets.compare_digest(password, settings.DOCS_PASSWORD)
                if not (valid_user and valid_pass):
                    return Response(
                        status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="HireRight AI Docs"'},
                        content="Unauthorized",
                    )
            return await call_next(request)

    app.add_middleware(CorrelationIdMiddleware)

    limiter = get_rate_limiter().limiter
    app.state.limiter = limiter

    async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        limit = exc.limit
        retry_after = limit.get_expiry() if hasattr(limit, "get_expiry") else 60
        correlation_id = getattr(request.state, "correlation_id", None)

        logger.error(
            "Rate limit exceeded",
            exc_info=exc,
            path=request.url.path,
            limit=str(limit.limit) if hasattr(limit, "limit") else None,
            correlation_id=correlation_id,
        )

        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": retry_after,
                "correlation_id": correlation_id,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-Correlation-Id": correlation_id or "",
            },
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
        ingest_queue = get_ingest_queue()
        dead_letter_count = ingest_queue.dead_letter_count()
        return {
            "status": "degraded" if dead_letter_count > 0 else "ok",
            "dead_letter_count": dead_letter_count,
        }

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
        logger.error("AI service unavailable", exc_info=exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "AI service temporarily unavailable"},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
