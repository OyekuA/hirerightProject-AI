from pydantic import BaseModel


class GenerateInviteEmailRequest(BaseModel):
    candidate_id: int
    candidate_version: int
    job_id: int
    job_version: int


class GenerateInviteEmailResponse(BaseModel):
    subject: str
    body: str
