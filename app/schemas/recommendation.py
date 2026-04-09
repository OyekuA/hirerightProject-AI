"""Pydantic schemas for the recommendation domain."""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class RecommendResult(BaseModel):
    """A single recommendation result."""
    id: int
    similarity_score: float
    llm_score: Optional[int] = Field(None, ge=0, le=100)


class RecentClick(BaseModel):
    """A single click event."""
    id: int
    dwell_time_seconds: int = 0


class BehavioralSignals(BaseModel):
    """Behavioral signals used for adaptive weighting."""
    recent_searches: list[str] = []
    recent_clicks: list[RecentClick] = []
    recent_saves: list[int] = []
    recent_positive_outcomes: list[int] = []


class RecommendRequest(BaseModel):
    """Request payload for generating recommendations."""
    type: Literal["jobs", "candidates"]
    target_id: int
    target_version: int
    behavioral_signals: BehavioralSignals = Field(default_factory=BehavioralSignals)
    hard_filters: dict = {}
    force_refresh: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    """Response payload containing the list of recommendations."""
    results: list[RecommendResult]


class PoolRankRequest(BaseModel):
    """Request payload for ranking a candidate pool."""
    job_id: int
    job_version: int
    candidate_ids: list[int] = Field(..., min_length=1, max_length=100)
    force_refresh: bool = False

    @field_validator('candidate_ids')
    @classmethod
    def validate_unique_candidate_ids(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('candidate_ids must be unique')
        return v


class PoolRankResult(BaseModel):
    """A single pool ranking result."""
    candidate_id: int
    fit_score: int = Field(..., ge=0, le=100)


class PoolRankResponse(BaseModel):
    """Response payload containing the ranked candidate pool."""
    results: list[PoolRankResult]
