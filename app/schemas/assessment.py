"""Pydantic schemas for the assessment domain."""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Literal


class CandidateContext(BaseModel):
    candidate_id: int
    target_role: str


class JobContext(BaseModel):
    job_id: int


class GenerateAssessmentRequest(BaseModel):
    candidate_context: Optional[CandidateContext] = None
    job_context: Optional[JobContext] = None
    num_questions: int = Field(default=3)
    question_type: Literal["single", "multiple_choice"] = "single"

    @model_validator(mode='after')
    def at_least_one_context(self) -> 'GenerateAssessmentRequest':
        if self.candidate_context is None and self.job_context is None:
            raise ValueError('At least one of candidate_context or job_context must be provided')
        return self


class GradeAssessmentRequest(BaseModel):
    questions: List[Union[str, 'MultipleChoiceQuestion']]
    answers: List[str]
    time_taken_seconds: int = Field(..., gt=0)

    @model_validator(mode='after')
    def validate_questions_answers(self) -> 'GradeAssessmentRequest':
        if len(self.questions) != len(self.answers):
            raise ValueError('Number of questions must match number of answers')
        return self


class MultipleChoiceQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

    @model_validator(mode='after')
    def validate_options(self) -> 'MultipleChoiceQuestion':
        if len(self.options) != 4:
            raise ValueError('Exactly 4 options are required')
        if self.correct_answer not in self.options:
            raise ValueError('correct_answer must be one of the provided options')
        return self


class GenerateAssessmentResponse(BaseModel):
    question_type: Literal["single", "multiple_choice"]
    questions: List[Union[str, MultipleChoiceQuestion]]


class AuthenticityFlag(BaseModel):
    is_suspicious: bool
    reason: str


class SkillBreakdownItem(BaseModel):
    category: str
    score: int = Field(..., ge=0, le=100)
    feedback: str


class GradeAssessmentResponse(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    skill_breakdown: List[SkillBreakdownItem] = Field(..., min_length=3, max_length=5)
    authenticity_flag: AuthenticityFlag
