from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import structlog

from app.clients.dependencies import get_qdrant_client, get_llm_client, get_cache_backend, get_rate_limiter
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.services.decision_service import DecisionService
from app.routers._rate_limit_keys import candidate_id_key

router = APIRouter(tags=["Decision"])

limiter = get_rate_limiter().limiter


from app.schemas.decision import (
    DecisionRequest,
    DecisionResponse,
)


@router.post("/decision", response_model=DecisionResponse)
@limiter.limit("500/hour")
@limiter.limit("20/minute")
@limiter.limit("100/hour", key_func=candidate_id_key)
async def decision(
    request: Request,
    req: DecisionRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = DecisionService(llm=llm, qdrant=qdrant, cache=cache)
    try:
        result = await asyncio.to_thread(
            service.decide,
            candidate_id=req.candidate_id,
            candidate_version=req.candidate_version,
            job_id=req.job_id,
            job_version=req.job_version,
            assessment_score=req.assessment_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Decision service temporarily unavailable",
        )
    return DecisionResponse.model_validate(result)
