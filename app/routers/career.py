from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from typing import List
import structlog

from app.clients.dependencies import get_qdrant_client, get_gemini_client, get_rate_limiter
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.career_service import CareerPathService

router = APIRouter(tags=["Career"])

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

class CareerPathItem(BaseModel):
    """A single suggested career path."""
    role: str
    match_percentage: int = Field(..., ge=0, le=100)
    reasoning: str


class AnalyzeCareerPathsRequest(BaseModel):
    """Request payload for career‑path analysis."""
    candidate_id: int


class AnalyzeCareerPathsResponse(BaseModel):
    """Response payload containing three suggested career paths."""
    paths: List[CareerPathItem]


# --- Route handler ---

@router.post("/analyze-career-paths", response_model=AnalyzeCareerPathsResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=_candidate_id_key)
async def analyze_career_paths(
    request: Request,
    req: AnalyzeCareerPathsRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Suggest three career paths based on the candidate's profile."""
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = CareerPathService(gemini=gemini, qdrant=qdrant)
    try:
        result = service.analyze_career_paths(candidate_id=req.candidate_id)
        paths = []
        for item in result:
            try:
                paths.append(CareerPathItem(**item))
            except ValidationError as e:
                raise GeminiUnavailableError(f"Malformed Gemini output: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Career path service temporarily unavailable",
        )
    return AnalyzeCareerPathsResponse(paths=paths)