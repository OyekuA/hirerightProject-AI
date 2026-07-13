from typing import Optional, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator
from datetime import datetime


class InterviewStartRequest(BaseModel):
    meeting_url: HttpUrl
    job_id: int
    candidate_id: int
    rubric: list[str]
    callback_url: HttpUrl
    join_at: Optional[str] = None
    bot_name: Optional[str] = None

    @field_validator("join_at")
    @classmethod
    def validate_iso8601(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except (ValueError, TypeError):
            raise ValueError(
                "join_at must be ISO 8601 (e.g. '2026-07-12T14:30:00Z') or omitted to join immediately"
            )


class InterviewStartAcceptedResponse(BaseModel):
    session_id: str


class InterviewSessionStatusResponse(BaseModel):
    session_id: str
    status: Literal["pending", "recording", "transcribing", "grading", "completed", "failed"]
    result: Optional[dict] = None
    candidate_id: int
    job_id: int
