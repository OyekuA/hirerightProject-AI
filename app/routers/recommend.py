from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal, Optional
import structlog

from app.clients.dependencies import get_qdrant_client, get_gemini_client, get_cache_backend, get_rate_limiter
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.services.recommendation_service import RecommendationService

router = APIRouter(tags=["Recommendation"])

limiter = get_rate_limiter().limiter


async def _target_id_key(request: Request) -> str:
    """Extract target_id from request body for rate limiting."""
    try:
        body = await request.json()
        target_id = body.get("target_id")
        if target_id is None:
            return "unknown"
        return f"target:{target_id}"
    except Exception:
        return "unknown"


# --- Request/Response models ---

class RecommendResult(BaseModel):
    """A single recommendation result."""
    id: int
    similarity_score: float
    llm_score: Optional[int] = Field(None, ge=0, le=100)


class RecentClick(BaseModel):
    """A single click event."""
    id: int
    dwell_time_seconds: int = 0


class BehavioralSignals(BaseModel):
    """Behavioral signals used for adaptive weighting."""
    recent_searches: list[str] = []
    recent_clicks: list[RecentClick] = []
    recent_saves: list[int] = []
    recent_positive_outcomes: list[int] = []


class RecommendRequest(BaseModel):
    """Request payload for generating recommendations."""
    type: Literal["jobs", "candidates"]
    target_id: int
    target_version: int
    behavioral_signals: BehavioralSignals = Field(default_factory=BehavioralSignals)
    hard_filters: dict = {}
    force_refresh: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    """Response payload containing the list of recommendations."""
    results: list[RecommendResult]


# --- Route handler ---

@router.post("/recommend", response_model=RecommendResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=_target_id_key)
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
        raw_results = service.recommend(
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
    results = [RecommendResult(**r) for r in raw_results]
    return RecommendResponse(results=results)