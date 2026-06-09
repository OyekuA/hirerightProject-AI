from fastapi import APIRouter, Depends, HTTPException, Request
import structlog
import asyncio

from app.clients.dependencies import get_llm_client, get_rate_limiter, get_qdrant_client
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.jd_service import JDService
from app.routers._rate_limit_keys import job_id_key

router = APIRouter(tags=["Job Description"])

limiter = get_rate_limiter().limiter


from app.schemas.jd import (
    GenerateJDRequest,
    GenerateJDResponse,
    AnalyzeJDRequest,
    AnalyzeJDResponse,
)


@router.post("/generate-jd", response_model=GenerateJDResponse)
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("50/day", key_func=job_id_key)
async def generate_jd(
    request: Request,
    req: GenerateJDRequest,
    llm: LLMClient = Depends(get_llm_client),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(llm=llm, qdrant=qdrant)
    try:
        jd_text = await asyncio.to_thread(
            service.generate_jd,
            prompt=req.prompt,
            existing_draft=req.existing_draft,
            job_id=req.job_id,
        )
    except LLMUnavailableError:
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
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("50/day", key_func=job_id_key)
async def analyze_jd(
    request: Request,
    req: AnalyzeJDRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(llm=llm)
    try:
        critiques = await asyncio.to_thread(service.analyze_jd, jd_text=req.jd_text)
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="JD analysis service temporarily unavailable",
        )
    return AnalyzeJDResponse(critiques=critiques)
