"""Pydantic schemas for the job description domain."""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class GenerateJDRequest(BaseModel):
    """Request payload for JD generation."""
    model_config = ConfigDict(extra='forbid')

    prompt: str
    existing_draft: Optional[str] = None
    job_id: Optional[int] = None


class GenerateJDResponse(BaseModel):
    """Response payload containing the generated job description."""
    jd_text: str


class AnalyzeJDRequest(BaseModel):
    """Request payload for JD analysis."""
    jd_text: str


class AnalyzeJDResponse(BaseModel):
    """Response payload containing a list of critique points."""
    critiques: List[str]

