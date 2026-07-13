from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import structlog

from app.clients.dependencies import get_qdrant_client, get_llm_client, get_rate_limiter
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.services.email_service import EmailGenerationService
from app.routers._rate_limit_keys import candidate_id_key

router = APIRouter(tags=["Email"])

limiter = get_rate_limiter().limiter


from app.schemas.email import (
    GenerateInviteEmailRequest,
    GenerateInviteEmailResponse,
)


@router.post("/generate-invite-email", response_model=GenerateInviteEmailResponse)
@limiter.limit("500/hour")
@limiter.limit("20/minute")
@limiter.limit("100/hour", key_func=candidate_id_key)
async def generate_invite_email(
    request: Request,
    req: GenerateInviteEmailRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    llm: LLMClient = Depends(get_llm_client),
):
    structlog.contextvars.bind_contextvars(entity_id=req.candidate_id)
    service = EmailGenerationService(llm=llm, qdrant=qdrant)
    try:
        result = await asyncio.to_thread(
            service.generate_invite_email,
            candidate_id=req.candidate_id,
            candidate_version=req.candidate_version,
            job_id=req.job_id,
            job_version=req.job_version,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Email generation service temporarily unavailable",
        )
    return GenerateInviteEmailResponse.model_validate(result)
