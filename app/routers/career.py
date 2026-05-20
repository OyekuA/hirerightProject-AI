from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
import structlog
import asyncio

from app.clients.dependencies import get_qdrant_client, get_llm_client, get_rate_limiter
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.career_service import CareerPathService
from app.routers._rate_limit_keys import candidate_id_key

router = APIRouter(tags=["Career"])

limiter = get_rate_limiter().limiter




from app.schemas.career import (
    CareerPathItem,
    AnalyzeCareerPathsRequest,
    AnalyzeCareerPathsResponse,
)

# --- Route handler ---

@router.post("/analyze-career-paths", response_model=AnalyzeCareerPathsResponse)
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("20/day", key_func=candidate_id_key)
async def analyze_career_paths(
    request: Request,
    req: AnalyzeCareerPathsRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
):
    """Suggest three career paths based on the candidate's profile."""
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = CareerPathService(llm=llm, qdrant=qdrant)
    try:
        result = await asyncio.to_thread(service.analyze_career_paths, candidate_id=req.candidate_id)
        paths = []
        for item in result["paths"]:
            try:
                paths.append(CareerPathItem(**item))
            except ValidationError as e:
                raise LLMUnavailableError(f"Malformed LLM output: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Career path service temporarily unavailable",
        )
    return AnalyzeCareerPathsResponse(profile_summary=result["profile_summary"], paths=paths)
