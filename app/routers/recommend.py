from fastapi import APIRouter, Depends, HTTPException, Request
import structlog
import asyncio

from app.clients.dependencies import get_qdrant_client, get_gemini_client, get_cache_backend, get_rate_limiter
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.services.recommendation_service import RecommendationService
from app.routers._rate_limit_keys import target_id_key

router = APIRouter(tags=["Recommendation"])

limiter = get_rate_limiter().limiter




from app.schemas.recommendation import (
    RecommendResult,
    RecentClick,
    BehavioralSignals,
    RecommendRequest,
    RecommendResponse,
    PoolRankRequest,
    PoolRankResult,
    PoolRankResponse,
)

# --- Route handler ---

@router.post("/recommend", response_model=RecommendResponse)
@limiter.limit("200/day")
@limiter.limit("20/hour", key_func=target_id_key)
async def recommend(
    request: Request,
    req: RecommendRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    """Generate recommendations for a target profile."""
    structlog.contextvars.bind_contextvars(entity_id=req.target_id)
    service = RecommendationService(gemini=gemini, qdrant=qdrant, cache=cache)
    try:
        raw_results = await asyncio.to_thread(
            service.recommend,
            rec_type=req.type,
            target_id=req.target_id,
            target_version=req.target_version,
            behavioral_signals=req.behavioral_signals.model_dump(),
            hard_filters=req.hard_filters,
            force_refresh=req.force_refresh,
            limit=req.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service temporarily unavailable",
        )
    results = [RecommendResult.model_validate(r) for r in raw_results]
    return RecommendResponse(results=results)


@router.post("/recommend/pool", response_model=PoolRankResponse)
@limiter.limit("100/hour")
async def pool_rank(
    request: Request,
    req: PoolRankRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    """Rank a pre‑filtered candidate pool by fit score."""
    structlog.contextvars.bind_contextvars(job_id=req.job_id)
    service = RecommendationService(gemini=gemini, qdrant=qdrant, cache=cache)
    try:
        raw_results = await asyncio.to_thread(
            service.rank_pool,
            job_id=req.job_id,
            job_version=req.job_version,
            candidate_ids=req.candidate_ids,
            force_refresh=req.force_refresh,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service temporarily unavailable",
        )
    results = [PoolRankResult.model_validate(r) for r in raw_results]
    return PoolRankResponse(results=results)