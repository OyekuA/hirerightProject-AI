"""Pydantic schemas for the ingestion domain."""

from pydantic import BaseModel, HttpUrl, field_validator
from typing import Literal, Optional, List


class ProfileData(BaseModel):
    name: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    candidate_version: int


class IngestCandidateRequest(BaseModel):
    candidate_id: int
    cv_url: HttpUrl
    profile_data: ProfileData
    callback_url: HttpUrl

    @field_validator("cv_url", "callback_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class JobMetadata(BaseModel):
    title: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    job_version: int
    company_name: Optional[str] = None
    about: Optional[str] = None


class CandidateExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    experience_level: Optional[str] = None
    industry: Optional[str] = None
    employment_type: Optional[str] = None
    skills: List[str] = []
    past_roles: List[str] = []
    raw_profile_summary: Optional[str] = None


class JobExtraction(BaseModel):
    title: Optional[str] = None
    location: Optional[str] = None
    experience_level: Optional[str] = None
    industry: Optional[str] = None
    employment_type: Optional[str] = None
    required_skills: List[str] = []
    raw_jd_summary: Optional[str] = None


class IngestJobRequest(BaseModel):
    job_id: int
    jd_text: str
    metadata: JobMetadata
    callback_url: HttpUrl

    @field_validator("callback_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class IngestionStatusResponse(BaseModel):
    event_id: str
    entity_type: Literal["candidate", "job"]
    entity_id: int
    status: Literal["pending", "running", "success", "failed"]
    attempt_count: int
    error_summary: Optional[str] = None
    callback_delivery_failed: bool = False
    created_at: str
    updated_at: str


class CVAutofillRequest(BaseModel):
    cv_url: HttpUrl

    @field_validator("cv_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v


class ExperienceEntry(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class EducationEntry(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class CVAutofillResponse(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    experience: List[ExperienceEntry] = []
    education: List[EducationEntry] = []
    certifications: List[str] = []
