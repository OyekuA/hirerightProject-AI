"""Background ingestion logic for candidates and jobs.

Implements the retry loop, Gemini extraction, embedding, Qdrant upsert,
and final status/callback update.
"""

import asyncio
import hashlib
import json
import structlog
from datetime import datetime, timezone

from app.config import get_settings
from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient
from app.clients.qdrant import QdrantClient
from app.services.ingestion_store import IngestionStatusStore
from app.services.ingestion_fetch import fetch_and_parse_cv
from app.services.callback_client import CallbackClient
from app.utils import truncate_to_prompt_cap, parse_gemini_json


logger = structlog.get_logger()
settings = get_settings()


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

    error_summary = None
    max_retries = 3
    total_attempts = max_retries + 1
    backoff_base = 2

    for attempt in range(total_attempts):
        try:
            cv_text = await asyncio.to_thread(fetch_and_parse_cv, cv_url)
            cv_text = truncate_to_prompt_cap(cv_text)
            new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
            existing_payload = qdrant.get(CANDIDATES_COLLECTION, candidate_id)
            if existing_payload is not None and existing_payload.get("cv_hash") == new_hash:
                qdrant.update_payload(CANDIDATES_COLLECTION, candidate_id, {
                    "candidate_version": profile_data.get("candidate_version", 1),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "cv_hash": new_hash
                })
                logger.info("Candidate ingestion skipped (hash match)", event_id=event_id, candidate_id=candidate_id)
                store.update(event_id, status="success", attempt_count=attempt + 1)
                error_summary = None
                break
            logger.debug("CV parsed", event_id=event_id, attempt=attempt + 1, text_length=len(cv_text))

            extraction_prompt = f"""
Extract the following fields from the CV and the provided profile data.
Return a valid JSON object with exactly these keys:
- "name": string
- "location": string
- "experience_level": string
- "industry": string
- "employment_type": string
- "skills": array of strings
- "past_roles": array of strings
- "raw_profile_summary": string (concise summary of the candidate's overall profile)

CV content:
{cv_text}

Profile data:
{json.dumps(profile_data, indent=2)}

Return only the JSON object, no other text.
"""
            generated = gemini.generate(extraction_prompt)
            extracted = parse_gemini_json(generated)
            raw_profile_summary = extracted.get("raw_profile_summary", "")
            if not raw_profile_summary:
                raw_profile_summary = generated[:500]
            raw_profile_summary = truncate_to_prompt_cap(raw_profile_summary)

            vector = gemini.embed(raw_profile_summary)

            payload = {
                "name": extracted.get("name") or profile_data.get("name", ""),
                "candidate_id": candidate_id,
                "location": extracted.get("location") or profile_data.get("location", ""),
                "experience_level": extracted.get("experience_level") or profile_data.get("experience_level", ""),
                "industry": extracted.get("industry") or profile_data.get("industry", ""),
                "employment_type": extracted.get("employment_type") or profile_data.get("employment_type", ""),
                "skills": extracted.get("skills") or [],
                "past_roles": extracted.get("past_roles") or [],
                "raw_profile_summary": raw_profile_summary,
                "candidate_version": profile_data.get("candidate_version", 1),
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
        delivered = callback_client.send(
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

    error_summary = None
    max_retries = 3
    total_attempts = max_retries + 1
    backoff_base = 2

    for attempt in range(total_attempts):
        try:
            jd_text = truncate_to_prompt_cap(jd_text)
            new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
            existing_payload = qdrant.get(JOBS_COLLECTION, job_id)
            if existing_payload is not None and existing_payload.get("jd_hash") == new_hash:
                qdrant.update_payload(JOBS_COLLECTION, job_id, {
                    "job_version": metadata.get("job_version", 1),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "jd_hash": new_hash
                })
                logger.info("Job ingestion skipped (hash match)", event_id=event_id, job_id=job_id)
                store.update(event_id, status="success", attempt_count=attempt + 1)
                error_summary = None
                break
            extraction_prompt = f"""
Extract the following fields from the job description.
Return a valid JSON object with exactly these keys:
- "title": string
- "location": string
- "experience_level": string
- "industry": string
- "employment_type": string
- "required_skills": array of strings
- "raw_jd_summary": string (concise summary of the job description)

Job description:
{jd_text}

Metadata:
{json.dumps(metadata, indent=2)}

Return only the JSON object, no other text.
"""
            generated = gemini.generate(extraction_prompt)
            extracted = parse_gemini_json(generated)
            raw_jd_summary = extracted.get("raw_jd_summary", "")
            if not raw_jd_summary:
                raw_jd_summary = generated[:500]
            raw_jd_summary = truncate_to_prompt_cap(raw_jd_summary)

            vector = gemini.embed(raw_jd_summary)

            payload = {
                "title": extracted.get("title") or metadata.get("title", ""),
                "job_id": job_id,
                "location": extracted.get("location") or metadata.get("location", ""),
                "experience_level": extracted.get("experience_level") or metadata.get("experience_level", ""),
                "industry": extracted.get("industry") or metadata.get("industry", ""),
                "employment_type": extracted.get("employment_type") or metadata.get("employment_type", ""),
                "required_skills": extracted.get("required_skills") or [],
                "raw_jd_summary": raw_jd_summary,
                "job_version": metadata.get("job_version", 1),
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
        delivered = callback_client.send(
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