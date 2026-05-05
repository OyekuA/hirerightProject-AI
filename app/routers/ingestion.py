from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Optional
import structlog

from app.utils.ingestion import validate_ingest_url
from app.services.ingestion_service import run_candidate_ingestion, run_job_ingestion
from app.services.ingestion_store import IngestionStatusStore
from app.services.callback_client import CallbackClient
from app.services.ingest_queue import IngestQueue
from app.clients.dependencies import (
    get_qdrant_client,
    get_llm_client,
    get_callback_client,
    get_ingestion_store,
    get_ingest_queue,
    get_rate_limiter,
    get_cache_backend,
    CANDIDATES_COLLECTION,
    JOBS_COLLECTION,
)
from app.clients.qdrant import QdrantClient
from app.clients.llm import LLMClient
from app.clients.cache import CacheBackend
from app.routers._rate_limit_keys import candidate_id_key, job_id_key

router = APIRouter(tags=["Ingestion"])

limiter = get_rate_limiter().limiter




from app.schemas.ingestion import (
    IngestCandidateRequest,
    IngestJobRequest,
    IngestionStatusResponse,
)

@router.post("/ingest-candidate")
@limiter.limit("200/day")
@limiter.limit("5/day", key_func=candidate_id_key)
async def ingest_candidate(
    request: Request,
    req: IngestCandidateRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    store: IngestionStatusStore = Depends(get_ingestion_store),
    callback_client: CallbackClient = Depends(get_callback_client),
    ingest_queue: IngestQueue = Depends(get_ingest_queue),
):
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    try:
        validate_ingest_url(str(req.cv_url))
        validate_ingest_url(str(req.callback_url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    record = store.create(
        entity_type="candidate",
        entity_id=req.candidate_id,
        callback_url=str(req.callback_url),
        payload={
            "cv_url": str(req.cv_url),
            "profile_data": req.profile_data.model_dump(),
        },
    )

    background_tasks.add_task(
        run_candidate_ingestion,
        candidate_id=req.candidate_id,
        cv_url=str(req.cv_url),
        profile_data=req.profile_data.model_dump(),
        callback_url=str(req.callback_url),
        event_id=record.event_id,
        qdrant=qdrant,
        llm=llm,
        store=store,
        callback_client=callback_client,
        ingest_queue=ingest_queue,
    )

    return JSONResponse(
        status_code=202,
        content={"event_id": record.event_id},
    )


@router.post("/ingest-job")
@limiter.limit("200/day")
@limiter.limit("5/day", key_func=job_id_key)
async def ingest_job(
    request: Request,
    req: IngestJobRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    store: IngestionStatusStore = Depends(get_ingestion_store),
    callback_client: CallbackClient = Depends(get_callback_client),
    ingest_queue: IngestQueue = Depends(get_ingest_queue),
):
    structlog.contextvars.bind_contextvars(entity_id=req.job_id)
    try:
        validate_ingest_url(str(req.callback_url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    record = store.create(
        entity_type="job",
        entity_id=req.job_id,
        callback_url=str(req.callback_url),
        payload={
            "jd_text": req.jd_text,
            "metadata": req.metadata.model_dump(),
        },
    )

    background_tasks.add_task(
        run_job_ingestion,
        job_id=req.job_id,
        jd_text=req.jd_text,
        metadata=req.metadata.model_dump(),
        callback_url=str(req.callback_url),
        event_id=record.event_id,
        qdrant=qdrant,
        llm=llm,
        store=store,
        callback_client=callback_client,
        ingest_queue=ingest_queue,
    )

    return JSONResponse(
        status_code=202,
        content={"event_id": record.event_id},
    )


@router.delete("/candidates/{candidate_id}")
@limiter.limit("200/day")
async def delete_candidate(
    request: Request,
    candidate_id: int,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    structlog.contextvars.bind_contextvars(entity_id=candidate_id)
    existing = qdrant.get(CANDIDATES_COLLECTION, candidate_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    qdrant.delete(CANDIDATES_COLLECTION, candidate_id)
    cache.delete_by_prefix(f"{candidate_id}:")
    return {"deleted": True}


@router.delete("/jobs/{job_id}")
@limiter.limit("200/day")
async def delete_job(
    request: Request,
    job_id: int,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    structlog.contextvars.bind_contextvars(entity_id=job_id)
    existing = qdrant.get(JOBS_COLLECTION, job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Job not found")
    qdrant.delete(JOBS_COLLECTION, job_id)
    cache.delete_by_job_id(job_id)
    return {"deleted": True}


@router.get("/ingestion-status")
@limiter.limit("200/day")
async def get_ingestion_status(
    request: Request,
    event_id: Optional[str] = Query(None, description="Event ID of ingestion record"),
    entity_type: Optional[str] = Query(None, description="Entity type: candidate or job"),
    entity_id: Optional[int] = Query(None, description="Entity ID"),
    store: IngestionStatusStore = Depends(get_ingestion_store),
):
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    if event_id:
        record = store.get_by_event_id(event_id)
        if not record:
            raise HTTPException(status_code=404, detail="Ingestion record not found")
        return IngestionStatusResponse.model_validate(record.to_dict())
    elif entity_type is not None and entity_id is not None:
        record = store.get_by_entity(entity_type, entity_id)  # type: ignore
        if not record:
            raise HTTPException(status_code=404, detail="Ingestion record not found")
        return IngestionStatusResponse.model_validate(record.to_dict())
    else:
        raise HTTPException(
            status_code=422,
            detail="Either event_id or both entity_type and entity_id must be provided",
        )