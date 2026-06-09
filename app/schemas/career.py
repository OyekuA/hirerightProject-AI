from pydantic import BaseModel, Field
from typing import List


class CareerPathItem(BaseModel):
    role: str
    match_percentage: int = Field(..., ge=0, le=100)
    core_skills: List[str] = Field(default_factory=list, min_length=0, max_length=5)
    reasoning: str


class AnalyzeCareerPathsRequest(BaseModel):
    candidate_id: int


class AnalyzeCareerPathsResponse(BaseModel):
    profile_summary: str
    paths: List[CareerPathItem]
