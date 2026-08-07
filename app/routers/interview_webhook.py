from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
import structlog

from app.config import get_settings
from app.clients.dependencies import get_meeting_bot_client, get_interview_session_store, get_callback_client, get_llm_client
from app.clients.meeting_bot import MeetingBotClient
from app.clients.llm import LLMClient
from app.services.interview_session_store import InterviewSessionStore
from app.services.callback_client import CallbackClient
from app.services.interview_service import grade_transcript
from app.utils.webhook_verification import verify_recall_webhook

logger = structlog.get_logger()

router = APIRouter(tags=["Interview Webhook"], include_in_schema=False)


@router.post("/interview/webhook", status_code=204)
async def interview_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    bot: MeetingBotClient = Depends(get_meeting_bot_client),
    store: InterviewSessionStore = Depends(get_interview_session_store),
    callback_client: CallbackClient = Depends(get_callback_client),
    llm: LLMClient = Depends(get_llm_client),
):
    raw_body = await request.body()
    headers = dict(request.headers)
    settings = get_settings()

    payload = verify_recall_webhook(raw_body, headers, settings.RECALL_AI_WEBHOOK_SECRET)
    if payload is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event = payload.get("event")
    if not isinstance(event, str):
        logger.warning("Recall webhook missing event type", payload=payload)
        return

    data = payload.get("data", {})
    bot_obj = data.get("bot", {}) if isinstance(data, dict) else {}
    bot_id = bot_obj.get("id") if isinstance(bot_obj, dict) else None

    if not bot_id:
        logger.warning("Recall webhook missing bot.id", payload=payload)
        return

    record = store.get_by_bot_id(bot_id)
    if record is None:
        logger.warning("Recall webhook: no session found for bot_id", bot_id=bot_id)
        return

    if record.status in ("completed", "failed"):
        logger.info(
            "Ignoring Recall webhook for terminal session",
            bot_id=bot_id,
            session_id=record.session_id,
            status=record.status,
            webhook_event=event,
        )
        return

    if event == "recording.done" and record.status in ("transcribing", "grading"):
        logger.info(
            "Ignoring duplicate recording.done event",
            bot_id=bot_id,
            session_id=record.session_id,
            status=record.status,
        )
        return

    if event == "transcript.done" and record.status == "grading":
        logger.info(
            "Ignoring duplicate transcript.done event",
            bot_id=bot_id,
            session_id=record.session_id,
            status=record.status,
        )
        return

    if event == "recording.done":
        recording_obj = data.get("recording", {}) if isinstance(data, dict) else {}
        recording_id = recording_obj.get("id") if isinstance(recording_obj, dict) else None
        if not recording_id:
            logger.warning("recording.done event missing recording.id", bot_id=bot_id)
            return

        async def _handle_recording_done(sid: str, rid: str, rec):
            try:
                transcript_id = await bot.create_transcript(rid)
                logger.info("Transcript creation scheduled", session_id=sid, transcript_id=transcript_id)
            except Exception as e:
                logger.error("Transcript creation failed", session_id=sid, error=str(e))
                store.update(sid, status="failed", result={"error": str(e)})
                try:
                    await callback_client.send(
                        callback_url=rec.callback_url,
                        event_id=sid,
                        entity_type="interview",
                        entity_id=rec.candidate_id,
                        status="failed",
                        error=f"transcript_creation_failed: {e}",
                    )
                except Exception as cb_err:
                    logger.error("Failed to send failure callback", session_id=sid, error=str(cb_err))

        store.update(record.session_id, recording_id=recording_id, status="transcribing")
        background_tasks.add_task(_handle_recording_done, record.session_id, recording_id, record)

    elif event == "transcript.done":
        transcript_obj = data.get("transcript", {}) if isinstance(data, dict) else {}
        transcript_id = transcript_obj.get("id") if isinstance(transcript_obj, dict) else None
        if not transcript_id:
            logger.warning("transcript.done event missing transcript.id", bot_id=bot_id)
            return

        async def _handle_transcript_done(sid: str, tid: str, rec):
            try:
                raw_turns = await bot.fetch_transcript(tid)
                grading_result = grade_transcript(
                    llm=llm,
                    rubric=rec.rubric,
                    raw_turns=raw_turns,
                )
                await callback_client.send(
                    callback_url=rec.callback_url,
                    event_id=sid,
                    entity_type="interview",
                    entity_id=rec.candidate_id,
                    status="completed",
                    error=None,
                    extra_payload={
                        "session_id": sid,
                        "grading_result": grading_result,
                    },
                )
                store.update(sid, status="completed", result=grading_result)
            except Exception as e:
                logger.error("Transcript processing failed", session_id=sid, error=str(e))
                store.update(sid, status="failed", result={"error": str(e)})
                await callback_client.send(
                    callback_url=rec.callback_url,
                    event_id=sid,
                    entity_type="interview",
                    entity_id=rec.candidate_id,
                    status="failed",
                    error=str(e),
                )

        store.update(record.session_id, status="grading")
        background_tasks.add_task(_handle_transcript_done, record.session_id, transcript_id, record)

    elif event == "transcript.failed":
        error_info = data.get("error", "transcript_failed") if isinstance(data, dict) else "transcript_failed"
        logger.error("Recall transcript.failed event", bot_id=bot_id, error=error_info)
        store.update(record.session_id, status="failed", result={"error": str(error_info)})
        try:
            await callback_client.send(
                callback_url=record.callback_url,
                event_id=record.session_id,
                entity_type="interview",
                entity_id=record.candidate_id,
                status="failed",
                error=str(error_info),
            )
        except Exception as cb_err:
            logger.error("Failed to send failure callback for transcript.failed", session_id=record.session_id, error=str(cb_err))

    else:
        logger.info("Unhandled Recall webhook event", webhook_event=event, bot_id=bot_id)
