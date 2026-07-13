import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import structlog

from app.utils.ingestion import validate_ingest_url, validate_callback_url
from app.services.screening_service import BulkScreeningService
from app.services.screening_store import BatchScreeningStore
from app.services.callback_client import CallbackClient
from app.clients.dependencies import (
    get_qdrant_client,
    get_llm_client,
    get_cache_backend,
    get_callback_client,
    get_screening_store,
    get_rate_limiter,
)
from app.clients.qdrant import QdrantClient
from app.clients.llm import LLMClient
from app.clients.cache import CacheBackend
from app.schemas.screening import (
    ScreenBatchRequest,
    ScreenBatchAcceptedResponse,
    ScreenBatchStatusResponse,
    ScreeningResultItem,
)

logger = structlog.get_logger()

router = APIRouter(tags=["Screening"])

limiter = get_rate_limiter().limiter


@router.post("/screen-batch")
@limiter.limit("20/day")
@limiter.limit("5/minute")
async def screen_batch(
    request: Request,
    req: ScreenBatchRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    cache: CacheBackend = Depends(get_cache_backend),
    store: BatchScreeningStore = Depends(get_screening_store),
    callback_client: CallbackClient = Depends(get_callback_client),
):
    structlog.contextvars.bind_contextvars(batch_size=len(req.candidates))

    for candidate in req.candidates:
        try:
            await asyncio.to_thread(validate_ingest_url, str(candidate.cv_url))
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid CV URL for candidate '{candidate.candidate_ref}': {e}",
            )

    if req.callback_url is not None:
        try:
            await asyncio.to_thread(validate_callback_url, str(req.callback_url))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    service = BulkScreeningService(llm=llm, qdrant=qdrant, cache=cache, callback_client=callback_client)
    try:
        job_payload = await service.resolve_job_payload(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    job_id = req.job_id or 0
    job_version = req.job_version or 1

    job_ref: dict
    if req.job_id is not None:
        job_ref = {"job_id": req.job_id, "job_version": req.job_version}
    else:
        job_ref = {"extracted_job": job_payload}

    record = store.create(
        total=len(req.candidates),
        job_ref=job_ref,
        callback_url=str(req.callback_url) if req.callback_url else None,
    )

    background_tasks.add_task(
        service.process_batch,
        batch_id=record.batch_id,
        job_payload=job_payload,
        job_id=job_id,
        job_version=job_version,
        candidates=req.candidates,
        store=store,
    )

    return JSONResponse(
        status_code=202,
        content=ScreenBatchAcceptedResponse(batch_id=record.batch_id).model_dump(),
    )


@router.get("/screen-batch/{batch_id}")
async def get_screen_batch_status(
    request: Request,
    batch_id: str,
    store: BatchScreeningStore = Depends(get_screening_store),
):
    structlog.contextvars.bind_contextvars(batch_id=batch_id)

    record = store.get_by_batch_id(batch_id)
    if not record:
        raise HTTPException(status_code=404, detail="Screening batch not found")

    results = record.results or []

    def _sort_key(item):
        fs = item.get("fit_score")
        if fs is None:
            return (1, 0)
        return (0, -fs)

    sorted_results = sorted(results, key=_sort_key)

    completed_count = sum(
        1 for r in results if r.get("status") in ("scored", "failed")
    )

    return ScreenBatchStatusResponse(
        batch_id=record.batch_id,
        status=record.status,
        total=record.total,
        completed_count=completed_count,
        results=[ScreeningResultItem.model_validate(r) for r in sorted_results],
    )
