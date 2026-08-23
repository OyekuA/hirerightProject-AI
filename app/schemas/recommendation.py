from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RecommendResult(BaseModel):
    id: int
    similarity_score: float


class RecentClick(BaseModel):
    id: int


class BehavioralSignals(BaseModel):
    recent_searches: list[str] = Field(default_factory=list, max_length=5)
    recent_clicks: list[RecentClick] = Field(default_factory=list, max_length=20)
    recent_saves: list[int] = Field(default_factory=list, max_length=20)
    recent_positive_outcomes: list[int] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _cap_total_cooc(self):
        # Contract: total co-occurrence ids across clicks/saves/positive_outcomes <= 20.
        # Keep the tail (most recent) across the three lists; log-free silent trim.
        clicks = list(self.recent_clicks)
        saves = list(self.recent_saves)
        pos = list(self.recent_positive_outcomes)
        total = len(clicks) + len(saves) + len(pos)
        if total > 20:
            combined = [("click", c) for c in clicks] + [("save", s) for s in saves] + [("pos", p) for p in pos]
            kept = combined[-20:]
            self.recent_clicks = [v for k, v in kept if k == "click"]
            self.recent_saves = [v for k, v in kept if k == "save"]
            self.recent_positive_outcomes = [v for k, v in kept if k == "pos"]
        return self


class RecommendRequest(BaseModel):
    type: Literal["jobs", "candidates"]
    target_id: int
    target_version: int
    behavioral_signals: BehavioralSignals = Field(default_factory=BehavioralSignals)
    hard_filters: dict = {}
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
