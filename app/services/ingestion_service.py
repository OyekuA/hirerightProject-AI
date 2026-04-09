"""Background ingestion logic for candidates and jobs.

Implements the retry loop, Gemini extraction, embedding, Qdrant upsert,
and final status/callback update.
"""

import asyncio
import hashlib
import structlog
from datetime import datetime, timezone

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient
from app.clients.qdrant import QdrantClient
from app.services.ingestion_store import IngestionStatusStore
from app.utils.ingestion import fetch_and_parse_cv, truncate_to_prompt_cap
from app.services.callback_client import CallbackClient
from app.utils import parse_gemini_json
from app.prompts import CV_EXTRACTION_PROMPT_TEMPLATE, JD_EXTRACTION_PROMPT_TEMPLATE
from app.schemas.ingestion import ProfileData, JobMetadata, CandidateExtraction, JobExtraction
from pydantic import ValidationError


logger = structlog.get_logger()


async def run_candidate_ingestion(
    candidate_id: int,
    cv_url: str,
    profile_data: dict,
    callback_url: str,
    event_id: str,
    qdrant: QdrantClient,
    gemini: GeminiClient,
    store: IngestionStatusStore,
    callback_client: CallbackClient,
) -> None:
    """Ingest a candidate CV, extract structured profile, embed, and store in Qdrant.

    Performs up to 4 attempts with exponential backoff (2s, 4s, 8s). Updates the status store
    and sends a callback upon completion.
    """
    logger.info("Started candidate ingestion", event_id=event_id, candidate_id=candidate_id)

    store.update(event_id, status="running")
    
    validated_profile = ProfileData.model_validate(profile_data)
    
    error_summary = None
    max_retries = 3
    total_attempts = max_retries + 1
    backoff_base = 2

    for attempt in range(total_attempts):
        try:
            cv_text = await asyncio.to_thread(fetch_and_parse_cv, cv_url)
            new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
            cv_text = truncate_to_prompt_cap(cv_text)
            existing_payload = qdrant.get(CANDIDATES_COLLECTION, candidate_id)
            if existing_payload is not None and existing_payload.get("cv_hash") == new_hash:
                qdrant.update_payload(CANDIDATES_COLLECTION, candidate_id, {
                    "name": validated_profile.name,
                    "location": validated_profile.location,
                    "experience_level": validated_profile.experience_level,
                    "industry": validated_profile.industry,
                    "employment_type": validated_profile.employment_type,
                    "candidate_version": validated_profile.candidate_version,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "cv_hash": new_hash
                })
                logger.info("Candidate ingestion skipped (hash match)", event_id=event_id, candidate_id=candidate_id)
                store.update(event_id, status="success", attempt_count=attempt + 1)
                error_summary = None
                break
            logger.debug("CV parsed", event_id=event_id, attempt=attempt + 1, text_length=len(cv_text))

            profile_data_json = validated_profile.model_dump_json(indent=2)
            extraction_prompt = CV_EXTRACTION_PROMPT_TEMPLATE.format(
                cv_text=cv_text,
                profile_data_json=profile_data_json
            )
            generated = await asyncio.to_thread(gemini.generate, extraction_prompt)
            extracted = parse_gemini_json(generated)
            
            # Validate extraction output
            try:
                validated_extraction = CandidateExtraction.model_validate(extracted)
            except ValidationError as e:
                logger.warning("Gemini extraction output failed validation, using fallback values",
                               event_id=event_id, errors=str(e))
                validated_extraction = CandidateExtraction()
            
            raw_profile_summary = validated_extraction.raw_profile_summary
            if not raw_profile_summary:
                raw_profile_summary = generated[:500]
            raw_profile_summary = truncate_to_prompt_cap(raw_profile_summary)
            
            vector = await asyncio.to_thread(gemini.embed, raw_profile_summary)

            payload = {
                "name": validated_extraction.name or validated_profile.name,
                "candidate_id": candidate_id,
                "location": validated_extraction.location or validated_profile.location,
                "experience_level": validated_extraction.experience_level or validated_profile.experience_level,
                "industry": validated_extraction.industry or validated_profile.industry,
                "employment_type": validated_extraction.employment_type or validated_profile.employment_type,
                "skills": validated_extraction.skills,
                "past_roles": validated_extraction.past_roles,
                "raw_profile_summary": raw_profile_summary,
                "candidate_version": validated_profile.candidate_version,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "cv_hash": new_hash,
            }

            qdrant.upsert(CANDIDATES_COLLECTION, candidate_id, vector, payload)

            store.update(event_id, status="success", attempt_count=attempt + 1)
            error_summary = None
            logger.info("Candidate ingestion succeeded", event_id=event_id, candidate_id=candidate_id)
            break

        except Exception as e:
            error_summary = f"{type(e).__name__}: {str(e)}"
            logger.warning(
                "Candidate ingestion attempt failed",
                event_id=event_id,
                attempt=attempt + 1,
                max_attempts=total_attempts,
                error=error_summary,
            )
            store.update(event_id, attempt_count=attempt + 1)

            if attempt == total_attempts - 1:
                store.update(
                    event_id,
                    status="failed",
                    error_summary=error_summary[:500]
                )
                logger.error("Candidate ingestion failed after all retries", event_id=event_id)
            else:
                backoff = backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff)
                continue

    try:
        delivered = await callback_client.send(
            callback_url=callback_url,
            event_id=event_id,
            entity_type="candidate",
            entity_id=candidate_id,
            status="success" if error_summary is None else "failed",
            error=error_summary,
        )
        if not delivered:
            store.update(event_id, callback_delivery_failed=True)
            logger.warning("Callback delivery failed", event_id=event_id)
    except Exception as e:
        logger.error("Callback dispatch failed unexpectedly", event_id=event_id, error=str(e))
        store.update(event_id, callback_delivery_failed=True)


async def run_job_ingestion(
    job_id: int,
    jd_text: str,
    metadata: dict,
    callback_url: str,
    event_id: str,
    qdrant: QdrantClient,
    gemini: GeminiClient,
    store: IngestionStatusStore,
    callback_client: CallbackClient,
) -> None:
    """Ingest a job description, extract structured metadata, embed, and store in Qdrant.

    Same retry pattern as candidate ingestion, but no CV fetch step.
    """
    logger.info("Started job ingestion", event_id=event_id, job_id=job_id)

    store.update(event_id, status="running")
    
    validated_metadata = JobMetadata.model_validate(metadata)
    
    error_summary = None
    max_retries = 3
    total_attempts = max_retries + 1
    backoff_base = 2

    new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
    jd_text = truncate_to_prompt_cap(jd_text)

    for attempt in range(total_attempts):
        try:
            existing_payload = qdrant.get(JOBS_COLLECTION, job_id)
            if existing_payload is not None and existing_payload.get("jd_hash") == new_hash:
                qdrant.update_payload(JOBS_COLLECTION, job_id, {
                    "title": validated_metadata.title,
                    "location": validated_metadata.location,
                    "experience_level": validated_metadata.experience_level,
                    "industry": validated_metadata.industry,
                    "employment_type": validated_metadata.employment_type,
                    "job_version": validated_metadata.job_version,
                    "company_name": validated_metadata.company_name,
                    "about": validated_metadata.about,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "jd_hash": new_hash
                })
                logger.info("Job ingestion skipped (hash match)", event_id=event_id, job_id=job_id)
                store.update(event_id, status="success", attempt_count=attempt + 1)
                error_summary = None
                break
            metadata_json = validated_metadata.model_dump_json(indent=2)
            extraction_prompt = JD_EXTRACTION_PROMPT_TEMPLATE.format(
                jd_text=jd_text,
                metadata_json=metadata_json
            )
            generated = await asyncio.to_thread(gemini.generate, extraction_prompt)
            extracted = parse_gemini_json(generated)
            
            # Validate extraction output
            try:
                validated_extraction = JobExtraction.model_validate(extracted)
            except ValidationError as e:
                logger.warning("Gemini extraction output failed validation, using fallback values",
                               event_id=event_id, errors=str(e))
                validated_extraction = JobExtraction()
            
            raw_jd_summary = validated_extraction.raw_jd_summary
            if not raw_jd_summary:
                raw_jd_summary = generated[:500]
            raw_jd_summary = truncate_to_prompt_cap(raw_jd_summary)

            vector = await asyncio.to_thread(gemini.embed, raw_jd_summary)

            payload = {
                "title": validated_extraction.title or validated_metadata.title,
                "job_id": job_id,
                "location": validated_extraction.location or validated_metadata.location,
                "experience_level": validated_extraction.experience_level or validated_metadata.experience_level,
                "industry": validated_extraction.industry or validated_metadata.industry,
                "employment_type": validated_extraction.employment_type or validated_metadata.employment_type,
                "required_skills": validated_extraction.required_skills,
                "raw_jd_summary": raw_jd_summary,
                "job_version": validated_metadata.job_version,
                "company_name": validated_metadata.company_name,
                "about": validated_metadata.about,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "jd_hash": new_hash,
            }

            qdrant.upsert(JOBS_COLLECTION, job_id, vector, payload)

            store.update(event_id, status="success", attempt_count=attempt + 1)
            error_summary = None
            logger.info("Job ingestion succeeded", event_id=event_id, job_id=job_id)
            break

        except Exception as e:
            error_summary = f"{type(e).__name__}: {str(e)}"
            logger.warning(
                "Job ingestion attempt failed",
                event_id=event_id,
                attempt=attempt + 1,
                max_attempts=total_attempts,
                error=error_summary,
            )
            store.update(event_id, attempt_count=attempt + 1)

            if attempt == total_attempts - 1:
                store.update(
                    event_id,
                    status="failed",
                    error_summary=error_summary[:500]
                )
                logger.error("Job ingestion failed after all retries", event_id=event_id)
            else:
                backoff = backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff)
                continue

    try:
        delivered = await callback_client.send(
            callback_url=callback_url,
            event_id=event_id,
            entity_type="job",
            entity_id=job_id,
            status="success" if error_summary is None else "failed",
            error=error_summary,
        )
        if not delivered:
            store.update(event_id, callback_delivery_failed=True)
            logger.warning("Callback delivery failed", event_id=event_id)
    except Exception as e:
        logger.error("Callback dispatch failed unexpectedly", event_id=event_id, error=str(e))
        store.update(event_id, callback_delivery_failed=True)