import json
import structlog
from typing import Optional

from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.clients.dependencies import JOBS_COLLECTION
from app.schemas.ingestion import JobMetadata
from app.services.ingestion_service import extract_job_entities
from app.utils import parse_llm_json, extract_list
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import JD_ANALYSIS_PROMPT_TEMPLATE, JD_GENERATION_PROMPT_TEMPLATE, JD_REFINEMENT_PROMPT_TEMPLATE

logger = structlog.get_logger()


class JDService:

    def __init__(self, llm: LLMClient, qdrant: Optional[QdrantClient] = None):
        self.llm = llm
        self.qdrant = qdrant

    @staticmethod
    def _build_context_blocks(context: dict) -> tuple[str, str, str]:
        comp_block = ""
        salary_min = context.get("salary_min")
        salary_max = context.get("salary_max")
        salary_currency = context.get("salary_currency")

        def _fmt(value) -> str:
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)

        if salary_min is not None or salary_max is not None:
            parts = []
            if salary_min is not None:
                parts.append(_fmt(salary_min))
            if salary_max is not None:
                parts.append(_fmt(salary_max))
            comp = " - ".join(parts)
            if salary_currency:
                comp = f"{comp} {salary_currency}"
            comp_block = f"Compensation Range: {comp}\n"

        benefits_block = ""
        benefits = context.get("benefits")
        if benefits:
            benefits_block = f"Benefits: {benefits}\n"

        work_mode_block = ""
        work_mode = context.get("work_mode")
        if work_mode:
            regions = context.get("remote_regions") or []
            if regions:
                work_mode_block = f"Work Mode: {work_mode} ({', '.join(regions)})\n"
            else:
                work_mode_block = f"Work Mode: {work_mode}\n"

        return comp_block, benefits_block, work_mode_block

    def _resolve_job_context(
        self,
        job_id: Optional[int] = None,
        job_metadata: Optional[JobMetadata] = None,
    ) -> dict:
        if job_id is not None:
            if self.qdrant is None:
                raise ValueError(
                    "job_id provided but Qdrant client is not configured; "
                    "enrichment cannot be performed"
                )
            payload = self.qdrant.get(JOBS_COLLECTION, job_id)
            if payload is MISSING or payload is None:
                raise ValueError(f"Job with ID {job_id} not found in Qdrant")
            required_skills = payload.get("required_skills", [])
            logger.info(
                "Job context resolved from Qdrant",
                job_id=job_id,
                has_title=bool(payload.get("title")),
                has_location=bool(payload.get("location")),
                has_skills=bool(required_skills),
            )
            return {
                "title": payload.get("title", "[Job Title]"),
                "location": payload.get("location", "[Location]"),
                "skills": ", ".join(required_skills) if required_skills else "[Required Skills]",
                "summary": payload.get("raw_jd_summary", ""),
                "company": payload.get("company_name") or "[Company Name]",
                "about": payload.get("about") or "[About the Company]",
                "benefits": payload.get("benefits"),
                "salary_min": payload.get("salary_min"),
                "salary_max": payload.get("salary_max"),
                "salary_currency": payload.get("salary_currency"),
                "work_mode": payload.get("work_mode"),
                "remote_regions": payload.get("remote_regions") or [],
                "source": "qdrant",
            }

        if job_metadata is None:
            logger.info(
                "No job context fetched",
                has_qdrant=self.qdrant is not None,
                job_id_provided=False,
            )
            return {
                "title": "[Job Title]",
                "location": "[Location]",
                "skills": "[Required Skills]",
                "summary": "",
                "company": "[Company Name]",
                "about": "[About the Company]",
                "benefits": None,
                "salary_min": None,
                "salary_max": None,
                "salary_currency": None,
                "work_mode": None,
                "remote_regions": [],
                "source": "none",
            }

        metadata = job_metadata.model_dump()
        skills = "[Required Skills]"
        summary = ""
        if metadata.get("description"):
            extraction = extract_job_entities(
                jd_text=metadata["description"],
                metadata_json=json.dumps(metadata),
                llm=self.llm,
            )
            required_skills = extraction.required_skills
            if required_skills:
                skills = ", ".join(required_skills)
            summary = extraction.raw_jd_summary or ""
            logger.info(
                "Job context derived from inline description via extraction LLM",
                has_skills=bool(required_skills),
                has_summary=bool(summary),
            )
        else:
            logger.info(
                "Inline job metadata provided without description — using placeholders",
            )

        return {
            "title": metadata.get("title") or "[Job Title]",
            "location": metadata.get("location") or "[Location]",
            "skills": skills,
            "summary": summary,
            "company": metadata.get("company_name") or "[Company Name]",
            "about": metadata.get("about") or "[About the Company]",
            "benefits": metadata.get("benefits"),
            "salary_min": metadata.get("salary_min"),
            "salary_max": metadata.get("salary_max"),
            "salary_currency": metadata.get("salary_currency"),
            "work_mode": metadata.get("work_mode"),
            "remote_regions": metadata.get("remote_regions") or [],
            "source": "inline",
        }

    def generate_jd(
        self,
        prompt: str,
        existing_draft: Optional[str] = None,
        job_id: Optional[int] = None,
        job_metadata: Optional[JobMetadata] = None,
    ) -> str:
        logger.info(
            "Generating JD",
            prompt_length=len(prompt),
            has_existing_draft=existing_draft is not None,
            job_id=job_id,
            has_job_metadata=job_metadata is not None,
        )

        context = self._resolve_job_context(job_id=job_id, job_metadata=job_metadata)
        resolved_title = context["title"]
        resolved_location = context["location"]
        resolved_skills = context["skills"]
        resolved_summary = context["summary"]
        resolved_company = context["company"]
        resolved_about = context["about"]
        comp_block, benefits_block, work_mode_block = self._build_context_blocks(context)

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
            source=context["source"],
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
                comp_block=comp_block,
                benefits_block=benefits_block,
                work_mode_block=work_mode_block,
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
                comp_block=comp_block,
                benefits_block=benefits_block,
                work_mode_block=work_mode_block,
                existing_draft=existing_draft,
                prompt=prompt,
            )

        try:
            jd_text = self.llm.generate(prompt_text, temperature=0.7)
        except LLMUnavailableError:
            logger.error("LLM call failed during JD generation")
            raise

        logger.info(
            "JD generated successfully",
            output_length=len(jd_text),
        )
        return jd_text

    def analyze_jd(self, jd_text: str) -> list[str]:
        logger.info(
            "Analyzing JD",
            jd_length=len(jd_text),
        )

        jd_text = truncate_to_prompt_cap(jd_text)

        prompt = JD_ANALYSIS_PROMPT_TEMPLATE.format(jd_text=jd_text)

        def _attempt(temp: float) -> list[str]:
            rf = LLMClient.get_response_format(self.llm._model, "array")
            generated = self.llm.generate(prompt, temperature=temp, response_format=rf)
            critiques = extract_list(parse_llm_json(generated))

            for i, item in enumerate(critiques):
                if not isinstance(item, str):
                    logger.error(
                        f"Item {i} is not a string",
                        item=item,
                    )
                    raise LLMUnavailableError(f"Item {i} is not a string")
            return critiques

        try:
            return _attempt(temp=0)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "First JD analysis parse failed, attempting repair retry",
                error=str(e),
            )
            try:
                return _attempt(temp=0.3)
            except (json.JSONDecodeError, ValueError) as e2:
                logger.error(
                    "Repair retry also failed for JD analysis",
                    error=str(e2),
                )
                raise LLMUnavailableError(
                    f"LLM returned malformed critique JSON: {e2}"
                )
