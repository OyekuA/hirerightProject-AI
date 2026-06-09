from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class GenerateJDRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt: str
    existing_draft: Optional[str] = None
    job_id: Optional[int] = None


class GenerateJDResponse(BaseModel):
    jd_text: str


class AnalyzeJDRequest(BaseModel):
    jd_text: str


class AnalyzeJDResponse(BaseModel):
    critiques: List[str]
