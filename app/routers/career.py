from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
import structlog
import asyncio

from app.clients.cache import CacheBackend
from app.clients.dependencies import get_qdrant_client, get_llm_client, get_rate_limiter, get_cache_backend
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.career_service import CareerPathService, MalformedLLMResponseError
from app.routers._rate_limit_keys import candidate_id_key

router = APIRouter(tags=["Career"])

limiter = get_rate_limiter().limiter

logger = structlog.get_logger()


from app.schemas.career import (
    CareerPathItem,
    AnalyzeCareerPathsRequest,
    AnalyzeCareerPathsResponse,
)


@router.post("/analyze-career-paths", response_model=AnalyzeCareerPathsResponse)
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("20/day", key_func=candidate_id_key)
async def analyze_career_paths(
    request: Request,
    req: AnalyzeCareerPathsRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
    cache: CacheBackend = Depends(get_cache_backend),
):
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = CareerPathService(llm=llm, qdrant=qdrant, cache=cache)
    try:
        result = await asyncio.to_thread(service.analyze_career_paths, candidate_id=req.candidate_id)
        paths = []
        for item in result["paths"]:
            core_skills = item.get("core_skills")
            if isinstance(core_skills, list) and len(core_skills) > 5:
                logger.warning(
                    "Truncating core_skills to schema max of 5",
                    candidate_id=req.candidate_id,
                    original_count=len(core_skills),
                )
                item["core_skills"] = core_skills[:5]
            try:
                paths.append(CareerPathItem(**item))
            except ValidationError as e:
                raise MalformedLLMResponseError(f"Malformed LLM output: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except MalformedLLMResponseError:
        raise HTTPException(status_code=502, detail="AI service returned an invalid response")
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Career path service temporarily unavailable",
        )
    return AnalyzeCareerPathsResponse(profile_summary=result["profile_summary"], paths=paths)
