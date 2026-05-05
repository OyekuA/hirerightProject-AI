from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import structlog

from app.clients.dependencies import get_qdrant_client, get_llm_client, get_cache_backend, get_rate_limiter
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.services.scoring_service import ScoringService
from app.routers._rate_limit_keys import candidate_id_key

router = APIRouter(tags=["Scoring"])

limiter = get_rate_limiter().limiter




from app.schemas.scoring import (
    CategoryStatus,
    CategoryBreakdown,
    CalculateFitRequest,
    CalculateFitResponse,
)

# --- Route handler ---

@router.post("/calculate-fit", response_model=CalculateFitResponse)
@limiter.limit("500/hour")
@limiter.limit("50/hour", key_func=candidate_id_key)
async def calculate_fit(
    request: Request,
    req: CalculateFitRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    """Calculate a detailed fit score between a candidate and a job."""
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = ScoringService(llm=llm, qdrant=qdrant, cache=cache)
    try:
        result = await asyncio.to_thread(
            service.calculate_fit,
            candidate_id=req.candidate_id,
            candidate_version=req.candidate_version,
            job_id=req.job_id,
            job_version=req.job_version,
            force_refresh=req.force_refresh,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Fit scoring service temporarily unavailable",
        )
    return CalculateFitResponse.model_validate(result)
