from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List
import structlog
from app.clients.dependencies import get_qdrant_client, get_gemini_client
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.assessment_service import AssessmentService

router = APIRouter(tags=["Assessment"])

from fastapi import Request
from app.clients.dependencies import get_rate_limiter
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

class GenerateAssessmentRequest(BaseModel):
    candidate_id: int
    target_role: str
    num_questions: int = Field(default=3)


class GradeAssessmentRequest(BaseModel):
    questions: List[str]
    answers: List[str]
    time_taken_seconds: int

    @model_validator(mode='after')
    def validate_questions_answers(self) -> 'GradeAssessmentRequest':
        if len(self.questions) != len(self.answers):
            raise ValueError('Number of questions must match number of answers')
        return self

class GenerateAssessmentResponse(BaseModel):
    questions: List[str]


class AuthenticityFlag(BaseModel):
    is_suspicious: bool
    reason: str


class GradeAssessmentResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    feedback: str
    authenticity_flag: AuthenticityFlag


# --- Route handlers ---

@router.post("/assessment/generate", response_model=GenerateAssessmentResponse)
@limiter.limit("200/day")
@limiter.limit("10/day", key_func=_candidate_id_key)
async def generate_assessment(
    request: Request,
    req: GenerateAssessmentRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Generate scenario‑based interview questions for a candidate."""
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = AssessmentService(gemini=gemini, qdrant=qdrant)
    try:
        questions = service.generate_questions(
            candidate_id=req.candidate_id,
            target_role=req.target_role,
            num_questions=req.num_questions,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Assessment generation service temporarily unavailable",
        )
    return GenerateAssessmentResponse(questions=questions)


@router.post("/assessment/grade", response_model=GradeAssessmentResponse)
@limiter.limit("200/day")
async def grade_assessment(
    request: Request,
    req: GradeAssessmentRequest,
    gemini: GeminiClient = Depends(get_gemini_client),
):
    """Grade candidate answers and produce a score, feedback, and authenticity flag."""
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = AssessmentService(gemini=gemini, qdrant=None)
    try:
        result = service.grade_answers(
            questions=req.questions,
            answers=req.answers,
            time_taken_seconds=req.time_taken_seconds,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )
    except GeminiUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Grading service temporarily unavailable",
        )
    return GradeAssessmentResponse(**result)