from pydantic import BaseModel, Field
from typing import Literal


class CategoryStatus(BaseModel):
    status: Literal["pass", "warning", "fail"]
    short_reason: str
    score: int = Field(..., ge=0, le=100)


class CategoryBreakdown(BaseModel):
    skills: CategoryStatus
    role_match: CategoryStatus
    experience: CategoryStatus
    location: CategoryStatus
    employment_type: CategoryStatus


class CalculateFitRequest(BaseModel):
    candidate_id: int
    candidate_version: int
    job_id: int
    job_version: int
    force_refresh: bool = False


class CalculateFitResponse(BaseModel):
    overall_score_percentage: int = Field(..., ge=0, le=100)
    category_breakdown: CategoryBreakdown
    skill_gap_analysis: str
