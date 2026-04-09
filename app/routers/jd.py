from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import structlog
import asyncio

from app.clients.dependencies import get_gemini_client, get_rate_limiter, get_qdrant_client
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.jd_service import JDService
from app.routers._rate_limit_keys import job_id_key

router = APIRouter(tags=["Job Description"])

limiter = get_rate_limiter().limiter


async def _unknown_key(request: Request) -> str:
    """Default key for endpoints without an identifiable actor."""
    return "unknown"


from app.schemas.jd import (
    GenerateJDRequest,
    GenerateJDResponse,
    AnalyzeJDRequest,
    AnalyzeJDResponse,
)

# --- Route handlers ---

@router.post("/generate-jd", response_model=GenerateJDResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=job_id_key)
async def generate_jd(
    request: Request,
    req: GenerateJDRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    """Generate or refine a job description."""
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(gemini=gemini, qdrant=qdrant)
    try:
        jd_text = await asyncio.to_thread(
            service.generate_jd,
            prompt=req.prompt,
            existing_draft=req.existing_draft,
            job_id=req.job_id,
        )
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="JD generation service temporarily unavailable",
        )
    except ValueError as e:
        if "Qdrant client is not configured" in str(e):
            status_code = 400
        else:
            status_code = 404
        raise HTTPException(
            status_code=status_code,
            detail=str(e),
        )
    return GenerateJDResponse(jd_text=jd_text)


@router.post("/analyze-jd", response_model=AnalyzeJDResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=job_id_key)
async def analyze_jd(
    request: Request,
    req: AnalyzeJDRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Analyze a job description and return critique points."""
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(gemini=gemini)
    try:
        critiques = await asyncio.to_thread(service.analyze_jd, jd_text=req.jd_text)
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="JD analysis service temporarily unavailable",
        )
    return AnalyzeJDResponse(critiques=critiques)