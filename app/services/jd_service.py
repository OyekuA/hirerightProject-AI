"""Job Description service for generating and analyzing job descriptions.

This module provides the JDService class that orchestrates LLM calls
to generate job descriptions from prompts and to critique existing JDs.
"""

import json
import structlog
from typing import Optional

from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.dependencies import JOBS_COLLECTION
from app.config import get_settings
from app.utils import parse_llm_json
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import JD_ANALYSIS_PROMPT_TEMPLATE, JD_GENERATION_PROMPT_TEMPLATE, JD_REFINEMENT_PROMPT_TEMPLATE

logger = structlog.get_logger()


class JDService:
    """Service that encapsulates LLM‑based JD generation and analysis."""

    def __init__(self, llm: LLMClient, qdrant: Optional[QdrantClient] = None):
        """Initialize the JD service.

        Args:
            llm: A configured LLMClient instance.
            qdrant: Optional Qdrant client for job context lookup.
        """
        self.llm = llm
        self.qdrant = qdrant

    def generate_jd(
        self,
        prompt: str,
        existing_draft: Optional[str] = None,
        job_id: Optional[int] = None,
    ) -> str:
        """Generate or refine a job description.

        Args:
            prompt: Textual guidance for the desired JD (e.g., "We need a senior
                backend engineer with Python and Kubernetes experience").
            existing_draft: Optional existing JD text to refine.
            job_id: Optional ID of an ingested job to fetch context from Qdrant.

        Returns:
            The generated or refined JD as a plain text string.

        Raises:
            LLMUnavailableError: If the LLM circuit breaker is open or the call fails.
            ValueError: If a job_id is provided but the job is not found in Qdrant.
        """
        logger.info(
            "Generating JD",
            prompt_length=len(prompt),
            has_existing_draft=existing_draft is not None,
        )

        resolved_title = "[Job Title]"
        resolved_location = "[Location]"
        resolved_skills = "[Required Skills]"
        resolved_summary = ""
        resolved_company = "[Company Name]"
        resolved_about = "[About the Company]"

        if job_id is not None:
            if self.qdrant is None:
                raise ValueError(
                    "job_id provided but Qdrant client is not configured; "
                    "enrichment cannot be performed"
                )
            payload = self.qdrant.get(JOBS_COLLECTION, job_id)
            if payload is None:
                raise ValueError(f"Job with ID {job_id} not found in Qdrant")
            resolved_title = payload.get("title", "[Job Title]")
            resolved_location = payload.get("location", "[Location]")
            required_skills = payload.get("required_skills", [])
            resolved_skills = ", ".join(required_skills) if required_skills else "[Required Skills]"
            resolved_summary = payload.get("raw_jd_summary", "")
            resolved_company = payload.get("company_name") or "[Company Name]"
            resolved_about = payload.get("about") or "[About the Company]"
            logger.info(
                "Job context resolved from Qdrant",
                job_id=job_id,
                has_title=resolved_title != "[Job Title]",
                has_location=resolved_location != "[Location]",
                has_skills=bool(required_skills),
            )
        else:
            logger.info(
                "No job context fetched",
                has_qdrant=self.qdrant is not None,
                job_id_provided=False,
            )

        if resolved_company != "[Company Name]":
            prompt = prompt.replace("[Company Name]", resolved_company)
            if existing_draft is not None:
                existing_draft = existing_draft.replace("[Company Name]", resolved_company)
        if resolved_about != "[About the Company]":
            prompt = prompt.replace("[About]", resolved_about)
            prompt = prompt.replace("[About the Company]", resolved_about)
            if existing_draft is not None:
                existing_draft = existing_draft.replace("[About]", resolved_about)
                existing_draft = existing_draft.replace("[About the Company]", resolved_about)

        logger.info(
            "Context resolved",
            title=resolved_title,
            location=resolved_location,
            skills=resolved_skills,
            company=resolved_company,
        )

        prompt = truncate_to_prompt_cap(prompt)
        if existing_draft is not None:
            existing_draft = truncate_to_prompt_cap(existing_draft)

        if existing_draft is None:
            prompt_text = JD_GENERATION_PROMPT_TEMPLATE.format(
                title=resolved_title,
                company=resolved_company,
                location=resolved_location,
                skills=resolved_skills,
                about=resolved_about,
                summary=resolved_summary,
                prompt=prompt,
            )
        else:
            prompt_text = JD_REFINEMENT_PROMPT_TEMPLATE.format(
                title=resolved_title,
                company=resolved_company,
                location=resolved_location,
                skills=resolved_skills,
                about=resolved_about,
                summary=resolved_summary,
                existing_draft=existing_draft,
                prompt=prompt,
            )

        try:
            jd_text = self.llm.generate(prompt_text)
        except Exception as e:
            logger.error("LLM call failed during JD generation", error=str(e))
            raise LLMUnavailableError(f"JD generation failed: {e}")

        logger.info(
            "JD generated successfully",
            output_length=len(jd_text),
        )
        return jd_text

    def analyze_jd(self, jd_text: str) -> list[str]:
        """Analyze a job description and return a list of critique points.

        Args:
            jd_text: The job description text to analyse.

        Returns:
            A list of strings, each being a concrete critique suggestion.

        Raises:
            LLMUnavailableError: If the LLM circuit breaker is open or the call fails,
                or if the response is malformed.
        """
        logger.info(
            "Analyzing JD",
            jd_length=len(jd_text),
        )

        jd_text = truncate_to_prompt_cap(jd_text)

        prompt = JD_ANALYSIS_PROMPT_TEMPLATE.format(jd_text=jd_text)

        generated = self.llm.generate(prompt)

        try:
            critiques = parse_llm_json(generated)
        except json.JSONDecodeError as e:
            logger.error(
                "LLM returned non‑JSON response",
                error=str(e),
            )
            raise LLMUnavailableError(
                f"LLM returned malformed critique JSON: {e}"
            )

        if not isinstance(critiques, list):
            logger.error(
                "LLM response is not a list",
                response_type=type(critiques),
            )
            raise LLMUnavailableError("LLM response is not a list")

        for i, item in enumerate(critiques):
            if not isinstance(item, str):
                logger.error(
                    f"Item {i} is not a string",
                    item=item,
                )
                raise LLMUnavailableError(f"Item {i} is not a string")

        logger.info(
            "JD analysis completed",
            num_critiques=len(critiques),
        )
        return critiques
