from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, Literal
import structlog

from app.services.ingestion_fetch import validate_ingest_url
from app.services.ingestion_service import run_candidate_ingestion, run_job_ingestion
from app.services.ingestion_store import IngestionStatusStore
from app.services.callback_client import CallbackClient
from app.clients.dependencies import (
    get_qdrant_client,
    get_gemini_client,
    get_callback_client,
    get_ingestion_store,
    get_rate_limiter,
    CANDIDATES_COLLECTION,
    JOBS_COLLECTION,
)
from app.clients.qdrant import QdrantClient
from app.clients.gemini import GeminiClient

router = APIRouter(tags=["Ingestion"])

limiter = get_rate_limiter().limiter


async def _candidate_id_key(request: Request) -> str:
    """Extract candidate_id from request body for rate limiting."""
    try:
        body = await request.json()
        candidate_id = body.get("candidate_id")
        if candidate_id is None:
            return "unknown"
        return f"candidate:{candidate_id}"
    except Exception:
        return "unknown"


async def _job_id_key(request: Request) -> str:
    """Extract job_id from request body for rate limiting."""
    try:
        body = await request.json()
        job_id = body.get("job_id")
        if job_id is None:
            return "unknown"
        return f"job:{job_id}"
    except Exception:
        return "unknown"


# ========== Pydantic models for ingestion API ==========

class ProfileData(BaseModel):
    name: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    candidate_version: int


class IngestCandidateRequest(BaseModel):
    candidate_id: int
    cv_url: HttpUrl
    profile_data: ProfileData
    callback_url: HttpUrl

    @field_validator("cv_url", "callback_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class JobMetadata(BaseModel):
    title: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    job_version: int


class IngestJobRequest(BaseModel):
    job_id: int
    jd_text: str
    metadata: JobMetadata
    callback_url: HttpUrl

    @field_validator("callback_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class IngestionStatusResponse(BaseModel):
    event_id: str
    entity_type: Literal["candidate", "job"]
    entity_id: int
    status: Literal["pending", "running", "success", "failed"]
    attempt_count: int
    error_summary: Optional[str] = None
    callback_delivery_failed: bool = False
    created_at: str
    updated_at: str


@router.post("/ingest-candidate")
@limiter.limit("200/day")
@limiter.limit("5/day", key_func=_candidate_id_key)
async def ingest_candidate(
    request: Request,
    req: IngestCandidateRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
    store: IngestionStatusStore = Depends(get_ingestion_store),
    callback_client: CallbackClient = Depends(get_callback_client),
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
    )

    background_tasks.add_task(
        run_candidate_ingestion,
        candidate_id=req.candidate_id,
        cv_url=str(req.cv_url),
        profile_data=req.profile_data.dict(),
        callback_url=str(req.callback_url),
        event_id=record.event_id,
        qdrant=qdrant,
        gemini=gemini,
        store=store,
        callback_client=callback_client,
    )

    return JSONResponse(
        status_code=202,
        content={"event_id": record.event_id},
    )


@router.post("/ingest-job")
@limiter.limit("200/day")
@limiter.limit("5/day", key_func=_job_id_key)
async def ingest_job(
    request: Request,
    req: IngestJobRequest,
    background_tasks: BackgroundTasks,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
    store: IngestionStatusStore = Depends(get_ingestion_store),
    callback_client: CallbackClient = Depends(get_callback_client),
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
    )

    background_tasks.add_task(
        run_job_ingestion,
        job_id=req.job_id,
        jd_text=req.jd_text,
        metadata=req.metadata.dict(),
        callback_url=str(req.callback_url),
        event_id=record.event_id,
        qdrant=qdrant,
        gemini=gemini,
        store=store,
        callback_client=callback_client,
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
):
    structlog.contextvars.bind_contextvars(entity_id=candidate_id)
    existing = qdrant.get(CANDIDATES_COLLECTION, candidate_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    qdrant.delete(CANDIDATES_COLLECTION, candidate_id)
    return {"deleted": True}


@router.delete("/jobs/{job_id}")
@limiter.limit("200/day")
async def delete_job(
    request: Request,
    job_id: int,
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    structlog.contextvars.bind_contextvars(entity_id=job_id)
    existing = qdrant.get(JOBS_COLLECTION, job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Job not found")
    qdrant.delete(JOBS_COLLECTION, job_id)
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
        return IngestionStatusResponse(**record.to_dict())
    elif entity_type is not None and entity_id is not None:
        record = store.get_by_entity(entity_type, entity_id)  # type: ignore
        if not record:
            raise HTTPException(status_code=404, detail="Ingestion record not found")
        return IngestionStatusResponse(**record.to_dict())
    else:
        raise HTTPException(
            status_code=422,
            detail="Either event_id or both entity_type and entity_id must be provided",
        )