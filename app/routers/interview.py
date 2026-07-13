import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional
import structlog

from app.clients.dependencies import get_meeting_bot_client, get_interview_session_store, get_rate_limiter
from app.clients.meeting_bot import MeetingBotClient
from app.services.interview_session_store import InterviewSessionStore, InterviewSessionRecord
from app.routers._rate_limit_keys import candidate_id_key
from app.utils.ingestion import validate_callback_url

logger = structlog.get_logger()

router = APIRouter(tags=["Interview"])

limiter = get_rate_limiter().limiter


from app.schemas.interview import (
    InterviewStartRequest,
    InterviewStartAcceptedResponse,
    InterviewSessionStatusResponse,
)


@router.post("/interview/start", status_code=202)
@limiter.limit("500/day")
@limiter.limit("10/minute")
@limiter.limit("20/day", key_func=candidate_id_key)
async def start_interview(
    request: Request,
    req: InterviewStartRequest,
    bot: MeetingBotClient = Depends(get_meeting_bot_client),
    store: InterviewSessionStore = Depends(get_interview_session_store),
):
    structlog.contextvars.bind_contextvars(candidate_id=req.candidate_id, job_id=req.job_id)
    try:
        await asyncio.to_thread(validate_callback_url, str(req.callback_url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    import uuid
    session_id = str(uuid.uuid4())

    try:
        bot_id = await bot.inject_bot(
            meeting_url=str(req.meeting_url),
            session_id=session_id,
            candidate_id=req.candidate_id,
            join_at=req.join_at,
            bot_name=req.bot_name,
        )
    except Exception as e:
        logger.error("Failed to inject meeting bot", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to start interview bot")

    record = store.create(
        candidate_id=req.candidate_id,
        job_id=req.job_id,
        rubric=req.rubric,
        callback_url=str(req.callback_url),
        bot_id=bot_id,
        session_id=session_id,
    )

    return InterviewStartAcceptedResponse(session_id=record.session_id)


@router.get("/interview/{session_id}", response_model=InterviewSessionStatusResponse)
async def get_interview_status(
    request: Request,
    session_id: str,
    store: InterviewSessionStore = Depends(get_interview_session_store),
):
    record = store.get_by_session_id(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return InterviewSessionStatusResponse(
        session_id=record.session_id,
        status=record.status,
        result=record.result,
        candidate_id=record.candidate_id,
        job_id=record.job_id,
    )


@router.delete("/interview/{session_id}")
async def delete_interview_session(
    request: Request,
    session_id: str,
    store: InterviewSessionStore = Depends(get_interview_session_store),
):
    record = store.get_by_session_id(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Interview session not found")
    store.delete(session_id)
    return {"deleted": True}
