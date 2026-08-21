from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RecommendResult(BaseModel):
    id: int
    similarity_score: float


class RecentClick(BaseModel):
    id: int


class BehavioralSignals(BaseModel):
    recent_searches: list[str] = []
    recent_clicks: list[RecentClick] = []
    recent_saves: list[int] = []
    recent_positive_outcomes: list[int] = []


class RecommendRequest(BaseModel):
    type: Literal["jobs", "candidates"]
    target_id: int
    target_version: int
    behavioral_signals: BehavioralSignals = Field(default_factory=BehavioralSignals)
    hard_filters: dict = {}
    force_refresh: bool = False
    limit: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    results: list[RecommendResult]


class PoolRankRequest(BaseModel):
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
    candidate_id: int
    fit_score: int = Field(..., ge=0, le=100)
    status: Literal["scored", "failed", "timeout"] = "scored"


class PoolRankResponse(BaseModel):
    results: list[PoolRankResult]
