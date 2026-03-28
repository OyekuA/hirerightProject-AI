from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import structlog

from app.clients.dependencies import get_gemini_client, get_rate_limiter
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.services.jd_service import JDService

router = APIRouter(tags=["Job Description"])

limiter = get_rate_limiter().limiter


async def _unknown_key(request: Request) -> str:
    """Default key for endpoints without an identifiable actor."""
    return "unknown"


# --- Request/Response models ---

class GenerateJDRequest(BaseModel):
    """Request payload for JD generation."""
    prompt: str
    existing_draft: Optional[str] = None


class GenerateJDResponse(BaseModel):
    """Response payload containing the generated job description."""
    jd_text: str


class AnalyzeJDRequest(BaseModel):
    """Request payload for JD analysis."""
    jd_text: str


class AnalyzeJDResponse(BaseModel):
    """Response payload containing a list of critique points."""
    critiques: List[str]


# --- Route handlers ---

@router.post("/generate-jd", response_model=GenerateJDResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=_unknown_key)
async def generate_jd(
    request: Request,
    req: GenerateJDRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Generate or refine a job description."""
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(gemini=gemini)
    try:
        jd_text = service.generate_jd(
            prompt=req.prompt,
            existing_draft=req.existing_draft,
        )
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="JD generation service temporarily unavailable",
        )
    return GenerateJDResponse(jd_text=jd_text)


@router.post("/analyze-jd", response_model=AnalyzeJDResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=_unknown_key)
async def analyze_jd(
    request: Request,
    req: AnalyzeJDRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Analyze a job description and return critique points."""
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = JDService(gemini=gemini)
    try:
        critiques = service.analyze_jd(jd_text=req.jd_text)
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="JD analysis service temporarily unavailable",
        )
    return AnalyzeJDResponse(critiques=critiques)