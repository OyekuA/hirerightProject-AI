from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Literal, Optional, List

from app.constants import EMPLOYMENT_TYPES, WORK_MODES


def _validate_employment_type(v: str) -> str:
    key = v.lower().strip()
    if key == "":
        return key
    if key not in EMPLOYMENT_TYPES:
        raise ValueError(
            f"employment_type must be one of: {', '.join(EMPLOYMENT_TYPES)}"
        )
    return key


def _validate_work_mode(v: str) -> str:
    key = v.lower().strip()
    if key == "":
        return key
    if key not in WORK_MODES:
        raise ValueError(f"work_mode must be one of: {', '.join(WORK_MODES)}")
    return key


class ProfileData(BaseModel):
    name: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    candidate_version: int
    data_source: Optional[str] = None
    total_years_experience: Optional[float] = None
    work_mode: Optional[str] = None
    headline: Optional[str] = None

    @field_validator("employment_type")
    @classmethod
    def normalize_employment(cls, v: str) -> str:
        return _validate_employment_type(v)

    @field_validator("work_mode")
    @classmethod
    def normalize_work_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_work_mode(v)


class IngestCandidateRequest(BaseModel):
    candidate_id: int
    cv_url: HttpUrl
    profile_data: ProfileData
    callback_url: HttpUrl

    @field_validator("cv_url")
    @classmethod
    def ensure_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme != "https":
            raise ValueError("URL scheme must be HTTPS")
        return v

    @field_validator("callback_url")
    @classmethod
    def ensure_http_or_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in ("http", "https"):
            raise ValueError("Callback URL scheme must be http or https")
        return v


class JobMetadata(BaseModel):
    title: str
    location: str
    experience_level: str
    industry: str
    employment_type: str
    job_version: Optional[int] = 1
    company_name: Optional[str] = None
    about: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    responsibilities: Optional[str] = None
    benefits: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    work_mode: Optional[str] = None
    remote_regions: Optional[List[str]] = None

    @field_validator("employment_type")
    @classmethod
    def normalize_employment(cls, v: str) -> str:
        return _validate_employment_type(v)

    @field_validator("work_mode")
    @classmethod
    def normalize_work_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return _validate_work_mode(v)


class CandidateExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    experience_level: Optional[str] = None
    industry: Optional[str] = None
    employment_type: Optional[str] = None
    work_mode: Optional[str] = None
    skills: List[str] = []
    past_roles: List[str] = []
    raw_profile_summary: Optional[str] = None
    total_years_experience: Optional[float] = None
    headline: Optional[str] = None


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
    jd_text: str = Field(..., max_length=500_000)
    metadata: JobMetadata
    callback_url: HttpUrl

    @field_validator("callback_url")
    @classmethod
    def ensure_http_or_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in ("http", "https"):
            raise ValueError("Callback URL scheme must be http or https")
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


class SocialLink(BaseModel):
    platform: Optional[str] = None
    url: Optional[str] = None


class CVAutofillResponse(BaseModel):
    name: Optional[str] = None
    bio: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None
    experience: List[ExperienceEntry] = []
    education: List[EducationEntry] = []
    certifications: List[str] = []
    social_links: List[SocialLink] = []
