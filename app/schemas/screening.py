from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from typing import Literal, Optional

from app.schemas.scoring import CategoryBreakdown
from app.schemas.ingestion import JobMetadata


class ScreeningCandidateInput(BaseModel):
    candidate_ref: str
    cv_url: HttpUrl

    @field_validator("cv_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class ScreenBatchRequest(BaseModel):
    job_id: Optional[int] = None
    job_version: Optional[int] = None
    jd_text: Optional[str] = Field(None, max_length=500_000)
    job_metadata: Optional[JobMetadata] = None
    candidates: list["ScreeningCandidateInput"] = Field(..., min_length=1)
    callback_url: Optional[HttpUrl] = None

    @field_validator("callback_url")
    @classmethod
    def ensure_http_or_https(cls, v: Optional[HttpUrl]) -> Optional[HttpUrl]:
        if v is not None and v.scheme not in ("http", "https"):
            raise ValueError("Callback URL scheme must be http or https")
        return v

    @model_validator(mode="after")
    def validate_job_source(self) -> "ScreenBatchRequest":
        from app.config import get_settings

        has_existing = self.job_id is not None or self.job_version is not None
        has_raw = self.jd_text is not None or self.job_metadata is not None

        if has_existing and has_raw:
            raise ValueError(
                "Provide either (job_id + job_version) for an existing job, "
                "or (jd_text + job_metadata) for a raw JD — not both"
            )
        if not has_existing and not has_raw:
            raise ValueError(
                "Provide either (job_id + job_version) for an existing job, "
                "or (jd_text + job_metadata) for a raw JD"
            )
        if self.job_id is not None and self.job_version is None:
            raise ValueError("job_version is required when job_id is provided")
        if self.job_version is not None and self.job_id is None:
            raise ValueError("job_id is required when job_version is provided")

        if self.jd_text is not None:
            if not self.jd_text.strip():
                raise ValueError("jd_text must not be empty in raw JD mode")
        elif has_raw and self.jd_text is None:
            raise ValueError(
                "jd_text is required when using raw JD mode; "
                "job_metadata alone is not sufficient"
            )
        settings = get_settings()
        if len(self.candidates) > settings.SCREENING_MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(self.candidates)} exceeds maximum "
                f"of {settings.SCREENING_MAX_BATCH_SIZE}"
            )

        return self


class ScreeningResultItem(BaseModel):
    candidate_ref: str
    status: Literal["scored", "failed"]
    fit_score: Optional[int] = Field(None, ge=0, le=100)
    category_breakdown: Optional[CategoryBreakdown] = None
    skill_gap_analysis: Optional[str] = None
    error: Optional[str] = None


class ScreenBatchAcceptedResponse(BaseModel):
    batch_id: str


class ScreenBatchStatusResponse(BaseModel):
    batch_id: str
    status: Literal["pending", "running", "completed", "failed"]
    total: int
    completed_count: int
    results: list[ScreeningResultItem]


