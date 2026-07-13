import json
import structlog

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.utils import parse_llm_json
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import EMAIL_GENERATION_PROMPT_TEMPLATE

logger = structlog.get_logger()


class EmailGenerationService:

    def __init__(self, llm: LLMClient, qdrant: QdrantClient):
        self.llm = llm
        self.qdrant = qdrant

    def generate_invite_email(
        self,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
    ) -> dict:
        logger.info(
            "Generating invite email",
            candidate_id=candidate_id,
            job_id=job_id,
        )

        candidate_payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if candidate_payload is MISSING or candidate_payload is None:
            logger.warning("Candidate not found", candidate_id=candidate_id)
            raise ValueError("Candidate not found")
        stored_candidate_version = candidate_payload.get("candidate_version", 1)
        if stored_candidate_version != candidate_version:
            logger.warning(
                "Candidate version mismatch",
                candidate_id=candidate_id,
                stored_version=stored_candidate_version,
                requested_version=candidate_version,
            )
            raise ValueError(
                f"Candidate version mismatch: stored {stored_candidate_version}, "
                f"requested {candidate_version}"
            )

        job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
        if job_payload is MISSING or job_payload is None:
            logger.warning("Job not found", job_id=job_id)
            raise ValueError("Job not found")
        stored_job_version = job_payload.get("job_version", 1)
        if stored_job_version != job_version:
            logger.warning(
                "Job version mismatch",
                job_id=job_id,
                stored_version=stored_job_version,
                requested_version=job_version,
            )
            raise ValueError(
                f"Job version mismatch: stored {stored_job_version}, "
                f"requested {job_version}"
            )

        candidate_name = candidate_payload.get("name", "Candidate")
        candidate_skills = candidate_payload.get("skills", [])
        candidate_summary = candidate_payload.get("raw_profile_summary", "")
        job_title = job_payload.get("title", "the position")
        company = job_payload.get("company_name", "our company")

        prompt = EMAIL_GENERATION_PROMPT_TEMPLATE.format(
            candidate_name=candidate_name,
            candidate_skills=", ".join(candidate_skills) if candidate_skills else "your skills",
            candidate_summary=candidate_summary or "your background",
            job_title=job_title,
            company=company,
        )
        prompt = truncate_to_prompt_cap(prompt)

        generated = self.llm.generate(prompt, temperature=0.7)
        result = self._parse_and_validate(generated)

        if not self._contains_calendar_link(result):
            logger.warning("Calendar link missing from email body, retrying once")
            generated = self.llm.generate(prompt, temperature=0.7)
            result = self._parse_and_validate(generated)
            if not self._contains_calendar_link(result):
                logger.error("Calendar link still missing after retry")
                raise LLMUnavailableError(
                    "LLM failed to include {{CALENDAR_LINK}} placeholder in email body after retry"
                )

        return result

    def _parse_and_validate(self, generated: str) -> dict:
        try:
            result = parse_llm_json(generated)
        except json.JSONDecodeError:
            raise LLMUnavailableError("LLM returned malformed JSON for email generation")
        if not isinstance(result, dict):
            raise LLMUnavailableError("LLM returned a non‑dict email payload")
        if "subject" not in result or "body" not in result:
            raise LLMUnavailableError("LLM response missing 'subject' or 'body'")
        if not isinstance(result["subject"], str) or not isinstance(result["body"], str):
            raise LLMUnavailableError("LLM response 'subject' and 'body' must be strings")
        return result

    @staticmethod
    def _contains_calendar_link(result: dict) -> bool:
        body = result.get("body", "")
        return "{{CALENDAR_LINK}}" in body
