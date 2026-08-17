from pydantic import BaseModel, Field
from typing import Literal


class DecisionRequest(BaseModel):
    candidate_id: int
    candidate_version: int
    job_id: int
    job_version: int
    assessment_score: int = Field(..., ge=0, le=100)
    needs_review: bool = False


class DecisionResponse(BaseModel):
    decision: Literal["hire", "no_hire", "review"]
    combined_score: int = Field(..., ge=0, le=100)
    fit_score: int = Field(..., ge=0, le=100)
    assessment_score: int = Field(..., ge=0, le=100)
    rationale: str
    confidence: int = Field(..., ge=0, le=100)
