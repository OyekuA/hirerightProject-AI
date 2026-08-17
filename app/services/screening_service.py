import asyncio
import structlog
from datetime import datetime, timezone
from typing import Optional

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.clients.cache import CacheBackend
from app.services.scoring_service import ScoringService
from app.services.screening_store import BatchScreeningStore
from app.services.callback_client import CallbackClient
from app.services.ingestion_service import extract_candidate_entities, extract_job_entities
from app.utils.ingestion import fetch_and_parse_cv, truncate_to_prompt_cap
from app.utils import parse_llm_json
from app.schemas.screening import ScreenBatchRequest
from app.schemas.ingestion import ProfileData, CandidateExtraction
from app.config import get_settings

logger = structlog.get_logger()


class BulkScreeningService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
        callback_client: Optional[CallbackClient] = None,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache
        self.callback_client = callback_client

    async def resolve_job_payload(
        self,
        req: ScreenBatchRequest,
    ) -> dict:
        settings = get_settings()

        if req.job_id is not None:
            job_payload = self.qdrant.get(JOBS_COLLECTION, req.job_id)
            if job_payload is MISSING or job_payload is None:
                raise ValueError(f"Job {req.job_id} not found in Qdrant")
            stored_version = job_payload.get("job_version", 1)
            if stored_version != req.job_version:
                raise ValueError(
                    f"Job version mismatch: stored {stored_version}, "
                    f"requested {req.job_version}"
                )
            return job_payload

        jd_text = req.jd_text  # type: ignore[assignment]
        metadata = req.job_metadata

        if metadata is not None:
            validated_metadata = metadata.model_dump()
            metadata_json = metadata.model_dump_json(indent=2)
        else:
            validated_metadata = {}
            metadata_json = "{}"

        jd_text = truncate_to_prompt_cap(jd_text)
        validated_extraction = await asyncio.to_thread(
            extract_job_entities, jd_text, metadata_json, self.llm,
        )

        job_payload = {
            "title": validated_extraction.title or validated_metadata.get("title"),
            "job_id": 0,
            "location": validated_extraction.location or validated_metadata.get("location"),
            "experience_level": validated_extraction.experience_level or validated_metadata.get("experience_level"),
            "industry": validated_extraction.industry or validated_metadata.get("industry"),
            "employment_type": validated_extraction.employment_type or validated_metadata.get("employment_type"),
            "required_skills": validated_extraction.required_skills,
            "raw_jd_summary": validated_extraction.raw_jd_summary or "",
            "job_version": validated_metadata.get("job_version", 1),
            "company_name": validated_metadata.get("company_name"),
            "about": validated_metadata.get("about"),
            "description": validated_metadata.get("description"),
            "requirements": validated_metadata.get("requirements"),
            "responsibilities": validated_metadata.get("responsibilities"),
            "benefits": validated_metadata.get("benefits"),
            "salary_min": validated_metadata.get("salary_min"),
            "salary_max": validated_metadata.get("salary_max"),
            "salary_currency": validated_metadata.get("salary_currency"),
            "work_mode": validated_metadata.get("work_mode"),
            "remote_regions": validated_metadata.get("remote_regions"),
        }

        return job_payload

    async def process_batch(
        self,
        batch_id: str,
        job_payload: dict,
        job_id: int,
        job_version: int,
        candidates: list,
        store: BatchScreeningStore,
    ) -> None:
        settings = get_settings()
        concurrency = settings.SCREENING_CONCURRENCY
        sem = asyncio.Semaphore(concurrency)

        store.update(batch_id, status="running")

        scoring_service = ScoringService(
            llm=self.llm,
            qdrant=self.qdrant,
            cache=self.cache,
        )

        breaker_open = False
        completed = 0

        async def _process_one(candidate_input) -> None:
            nonlocal breaker_open, completed

            async with sem:
                if breaker_open:
                    result = {
                        "candidate_ref": candidate_input.candidate_ref,
                        "status": "failed",
                        "fit_score": None,
                        "category_breakdown": None,
                        "skill_gap_analysis": None,
                        "error": "circuit breaker open — LLM unavailable",
                    }
                    store.append_result(batch_id, result)
                    return

                try:
                    cv_url = str(candidate_input.cv_url)
                    cv_text = await asyncio.to_thread(fetch_and_parse_cv, cv_url)
                    cv_text = truncate_to_prompt_cap(cv_text)

                    profile_data = {
                        "name": "",
                        "location": "",
                        "experience_level": "",
                        "industry": "",
                        "employment_type": "",
                        "candidate_version": 1,
                    }
                    validated_profile = ProfileData.model_validate(profile_data)
                    profile_data_json = validated_profile.model_dump_json(indent=2)

                    validated_extraction = await asyncio.to_thread(
                        extract_candidate_entities,
                        cv_text,
                        profile_data_json,
                        self.llm,
                    )

                    candidate_payload = {
                        "name": validated_extraction.name or validated_profile.name,
                        "candidate_id": 0,
                        "location": validated_extraction.location or validated_profile.location,
                        "experience_level": validated_extraction.experience_level or validated_profile.experience_level,
                        "industry": validated_extraction.industry or validated_profile.industry,
                        "employment_type": validated_extraction.employment_type or validated_profile.employment_type,
                        "skills": validated_extraction.skills,
                        "past_roles": validated_extraction.past_roles,
                        "raw_profile_summary": validated_extraction.raw_profile_summary or "",
                        "candidate_version": 1,
                    }

                    score_result = scoring_service.score_from_payloads(
                        candidate_payload=candidate_payload,
                        job_payload=job_payload,
                        candidate_id=0,
                        candidate_version=1,
                        job_id=job_id or 0,
                        job_version=job_version or 1,
                    )

                    result = {
                        "candidate_ref": candidate_input.candidate_ref,
                        "status": "scored",
                        "fit_score": score_result["overall_score_percentage"],
                        "category_breakdown": score_result["category_breakdown"],
                        "skill_gap_analysis": score_result["skill_gap_analysis"],
                        "error": None,
                    }

                except (ValueError, RuntimeError) as e:
                    result = {
                        "candidate_ref": candidate_input.candidate_ref,
                        "status": "failed",
                        "fit_score": None,
                        "category_breakdown": None,
                        "skill_gap_analysis": None,
                        "error": f"{type(e).__name__}: {str(e)[:500]}",
                    }
                except LLMUnavailableError as e:
                    error_str = str(e)
                    if "circuit breaker is open" in error_str.lower():
                        breaker_open = True
                    result = {
                        "candidate_ref": candidate_input.candidate_ref,
                        "status": "failed",
                        "fit_score": None,
                        "category_breakdown": None,
                        "skill_gap_analysis": None,
                        "error": f"LLMUnavailableError: {error_str[:500]}",
                    }

                store.append_result(batch_id, result)
                completed += 1

        try:
            tasks = [_process_one(c) for c in candidates]
            await asyncio.gather(*tasks)
        except Exception as exc:
            logger.error(
                "Screening batch failed with unexpected error",
                batch_id=batch_id,
                error=str(exc),
            )
            try:
                store.update(batch_id, status="failed", error_summary=str(exc)[:500])
            except Exception:
                logger.error(
                    "Failed to persist screening batch failure status",
                    batch_id=batch_id,
                )
            return

        store.update(batch_id, status="completed")

        if self.callback_client is not None:
            record = store.get_by_batch_id(batch_id)
            if record and record.callback_url:
                try:
                    entity_id = int(batch_id[:8], 16)
                except ValueError:
                    entity_id = hash(batch_id) % (2**31)

                try:
                    delivered = await self.callback_client.send(
                        callback_url=record.callback_url,
                        event_id=batch_id,
                        entity_type="screening_batch",
                        entity_id=entity_id,
                        status="success",
                        error=None,
                    )
                    if not delivered:
                        logger.warning(
                            "Screening callback delivery failed",
                            batch_id=batch_id,
                        )
                except Exception as e:
                    logger.error(
                        "Screening callback dispatch failed unexpectedly",
                        batch_id=batch_id,
                        error=str(e),
                    )

        logger.info(
            "Screening batch completed",
            batch_id=batch_id,
            total=len(candidates),
            completed_count=completed,
        )
