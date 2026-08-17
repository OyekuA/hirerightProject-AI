from pydantic import BaseModel, ConfigDict, model_validator
from typing import Optional, List

from app.schemas.ingestion import JobMetadata


class GenerateJDRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt: str
    existing_draft: Optional[str] = None
    job_id: Optional[int] = None
    job_metadata: Optional[JobMetadata] = None

    @model_validator(mode="after")
    def validate_job_source(self) -> "GenerateJDRequest":
        if self.job_id is not None and self.job_metadata is not None:
            raise ValueError(
                "Provide either job_id for an existing job or job_metadata "
                "for an inline job, not both"
            )
        return self


class GenerateJDResponse(BaseModel):
    jd_text: str


class AnalyzeJDRequest(BaseModel):
    jd_text: str


class AnalyzeJDResponse(BaseModel):
    critiques: List[str]
