from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal
import structlog

from app.clients.dependencies import get_qdrant_client, get_gemini_client, get_cache_backend, get_rate_limiter
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"])

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


# --- Request/Response models ---

class CategoryStatus(BaseModel):
    """Status and short reason for a single fit category."""
    status: Literal["pass", "warning", "fail"]
    short_reason: str


class CategoryBreakdown(BaseModel):
    """Breakdown of fit across four categories."""
    role_match: CategoryStatus
    experience: CategoryStatus
    location: CategoryStatus
    employment_type: CategoryStatus


class CalculateFitRequest(BaseModel):
    """Request payload for fit‑score calculation."""
    candidate_id: int
    candidate_version: int
    job_id: int
    job_version: int
    force_refresh: bool = False


class CalculateFitResponse(BaseModel):
    """Response payload containing the detailed fit score."""
    overall_score_percentage: int = Field(..., ge=0, le=100)
    category_breakdown: CategoryBreakdown
    skill_gap_analysis: str


# --- Route handler ---

@router.post("/calculate-fit", response_model=CalculateFitResponse)
@limiter.limit("500/hour")
@limiter.limit("50/hour", key_func=_candidate_id_key)
async def calculate_fit(
    request: Request,
    req: CalculateFitRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    """Calculate a detailed fit score between a candidate and a job."""
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = ScoringService(gemini=gemini, qdrant=qdrant, cache=cache)
    try:
        result = service.calculate_fit(
            candidate_id=req.candidate_id,
            candidate_version=req.candidate_version,
            job_id=req.job_id,
            job_version=req.job_version,
            force_refresh=req.force_refresh,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Fit scoring service temporarily unavailable",
        )
    return CalculateFitResponse(**result)