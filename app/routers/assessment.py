from fastapi import APIRouter, Depends, HTTPException
import structlog
import asyncio
from app.clients.dependencies import get_qdrant_client, get_llm_client
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.assessment_service import AssessmentService
from app.routers._rate_limit_keys import candidate_or_job_id_key

router = APIRouter(tags=["Assessment"])

from fastapi import Request
from app.clients.dependencies import get_rate_limiter
limiter = get_rate_limiter().limiter


from app.schemas.assessment import (
    GenerateAssessmentRequest,
    GradeAssessmentRequest,
    GenerateAssessmentResponse,
    GradeAssessmentResponse,
)


@router.post("/assessment/generate", response_model=GenerateAssessmentResponse)
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("50/day", key_func=candidate_or_job_id_key)
async def generate_assessment(
    request: Request,
    req: GenerateAssessmentRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
):
    candidate_id = req.candidate_context.candidate_id if req.candidate_context else None
    target_role = req.candidate_context.target_role if req.candidate_context else None
    job_id = req.job_context.job_id if req.job_context else None
    structlog.contextvars.bind_contextvars(entity_id=candidate_id)
    service = AssessmentService(llm=llm, qdrant=qdrant)
    try:
        questions = await asyncio.to_thread(
            service.generate_questions,
            candidate_id=candidate_id,
            target_role=target_role,
            num_questions=req.num_questions,
            job_id=job_id,
            question_type=req.question_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Assessment generation service temporarily unavailable",
        )
    return GenerateAssessmentResponse(question_type=req.question_type, questions=questions)


@router.post("/assessment/grade", response_model=GradeAssessmentResponse)
@limiter.limit("1000/day")
@limiter.limit("30/minute")
@limiter.limit("100/day", key_func=candidate_or_job_id_key)
async def grade_assessment(
    request: Request,
    req: GradeAssessmentRequest,
    llm: LLMClient = Depends(get_llm_client),
):
    structlog.contextvars.bind_contextvars(entity_id="unknown")
    service = AssessmentService(llm=llm, qdrant=None)
    try:
        result = await asyncio.to_thread(
            service.grade_answers,
            questions=req.questions,
            answers=req.answers,
            time_taken_seconds=req.time_taken_seconds,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Grading service temporarily unavailable",
        )
    return GradeAssessmentResponse.model_validate(result)
