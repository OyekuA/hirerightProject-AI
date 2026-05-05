"""Pydantic schemas for the career path domain."""

from pydantic import BaseModel, Field
from typing import List


class CareerPathItem(BaseModel):
    """A single suggested career path."""
    role: str
    match_percentage: int = Field(..., ge=0, le=100)
    core_skills: List[str] = Field(default_factory=list, min_length=0, max_length=5)
    reasoning: str


class AnalyzeCareerPathsRequest(BaseModel):
    """Request payload for career‑path analysis."""
    candidate_id: int


class AnalyzeCareerPathsResponse(BaseModel):
    """Response payload containing three suggested career paths."""
    profile_summary: str
    paths: List[CareerPathItem]
