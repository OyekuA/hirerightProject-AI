"""Pydantic schemas for the scoring domain."""

from pydantic import BaseModel, Field
from typing import Literal


class CategoryStatus(BaseModel):
    """Status and short reason for a single fit category."""
    status: Literal["pass", "warning", "fail"]
    short_reason: str


class CategoryBreakdown(BaseModel):
    """Breakdown of fit across four categories."""
    role_match: CategoryStatus
    experience: CategoryStatus
    location: CategoryStatus
    employment_type: CategoryStatus


class CalculateFitRequest(BaseModel):
    """Request payload for fit‑score calculation."""
    candidate_id: int
    candidate_version: int
    job_id: int
    job_version: int
    force_refresh: bool = False


class CalculateFitResponse(BaseModel):
    """Response payload containing the detailed fit score."""
    overall_score_percentage: int = Field(..., ge=0, le=100)
    category_breakdown: CategoryBreakdown
    skill_gap_analysis: str
