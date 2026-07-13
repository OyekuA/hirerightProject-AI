import asyncio
import hashlib
import json
import re
import structlog
from datetime import datetime, timezone, date
from typing import Callable, Awaitable, Optional, List, Tuple

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.services.ingestion_store import IngestionStatusStore
from app.utils.ingestion import fetch_and_parse_cv, truncate_to_prompt_cap
from app.services.callback_client import CallbackClient
from app.services.ingest_queue import IngestQueue
from app.config import get_settings
from app.utils import parse_llm_json
from app.prompts import CV_EXTRACTION_PROMPT_TEMPLATE, JD_EXTRACTION_PROMPT_TEMPLATE
from app.schemas.ingestion import ProfileData, JobMetadata, CandidateExtraction, JobExtraction
from pydantic import ValidationError


logger = structlog.get_logger()


_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_month_name(value: str) -> Optional[int]:
    """Parse a month name like 'Jan' or 'January' to month number (1-12)."""
    if not value:
        return None
    return _MONTH_NAMES.get(value.strip()[:3].lower())


_DATE_RANGE_PATTERN = re.compile(
    r"(?:(?P<start_month>[A-Za-z]+)\s+)?(?P<start_year>\d{4})\s*(?:–|-|to)\s*"
    r"(?:(?P<end_month>[A-Za-z]+)\s+)?(?:(?P<end_year>\d{4})|Present|Current|Now)",
    re.IGNORECASE,
)


def _compute_total_years_experience(past_roles: List[str]) -> Optional[float]:
    current_date_val = date.today()
    intervals_months: List[Tuple[int, int]] = []

    for role in past_roles:
        match = _DATE_RANGE_PATTERN.search(role)
        if not match:
            continue

        start_month_name = match.group("start_month")
        start_year_str = match.group("start_year")
        end_month_name = match.group("end_month")
        end_year_str = match.group("end_year")

        if not start_year_str:
            continue

        start_year = int(start_year_str)
        start_month = _parse_month_name(start_month_name)
        if start_month is None:
            start_month = 1

        if end_year_str:
            end_year = int(end_year_str)
            end_month = _parse_month_name(end_month_name)
            if end_month is None:
                end_month = 12
        else:
            end_year = current_date_val.year
            end_month = current_date_val.month

        if end_year < start_year or (end_year == start_year and end_month < start_month):
            continue

        start_total = start_year * 12 + start_month
        end_total = end_year * 12 + end_month + 1

        intervals_months.append((start_total, end_total))

    if not intervals_months:
        return None

    intervals_months.sort(key=lambda x: x[0])

    merged = [intervals_months[0]]
    for start, end in intervals_months[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    total_months = sum(end - start for start, end in merged)
    return round(total_months / 12.0, 2)


def extract_candidate_entities(
    cv_text: str,
    profile_data_json: str,
    llm: LLMClient,
) -> CandidateExtraction:
    extraction_prompt = CV_EXTRACTION_PROMPT_TEMPLATE.format(
        cv_text=cv_text,
        profile_data_json=profile_data_json,
    )

    def _attempt(temp: float) -> CandidateExtraction:
        generated = llm.generate(
            extraction_prompt,
            temperature=temp,
            response_format={"type": "json_object"},
        )
        extracted = parse_llm_json(generated)

        try:
            validated_extraction = CandidateExtraction.model_validate(extracted)
        except ValidationError as e:
            logger.warning(
                "LLM extraction output failed validation, using fallback values",
                errors=str(e),
            )
            validated_extraction = CandidateExtraction()
        raw_profile_summary = validated_extraction.raw_profile_summary
        if not raw_profile_summary:
            raw_profile_summary = generated[:500]
        object.__setattr__(validated_extraction, "raw_profile_summary", truncate_to_prompt_cap(raw_profile_summary))

        computed = _compute_total_years_experience(validated_extraction.past_roles)
        if computed is not None:
            object.__setattr__(validated_extraction, "total_years_experience", computed)

        return validated_extraction

    try:
        return _attempt(temp=0)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "First extraction parse failed, attempting repair retry",
            error=str(e),
        )
        try:
            return _attempt(temp=0.3)
        except (json.JSONDecodeError, ValueError) as e2:
            logger.error(
                "Repair retry also failed for extraction",
                error=str(e2),
            )
            raise LLMUnavailableError(f"LLM returned malformed extraction response: {e2}")


def extract_job_entities(
    jd_text: str,
    metadata_json: str,
    llm: LLMClient,
) -> JobExtraction:
    extraction_prompt = JD_EXTRACTION_PROMPT_TEMPLATE.format(
        jd_text=jd_text,
        metadata_json=metadata_json,
    )

    def _attempt(temp: float) -> JobExtraction:
        generated = llm.generate(
            extraction_prompt,
            temperature=temp,
            response_format={"type": "json_object"},
        )
        extracted = parse_llm_json(generated)

        try:
            validated_extraction = JobExtraction.model_validate(extracted)
        except ValidationError as e:
            logger.warning(
                "LLM extraction output failed validation, using fallback values",
                errors=str(e),
            )
            validated_extraction = JobExtraction()
        raw_jd_summary = validated_extraction.raw_jd_summary
        if not raw_jd_summary:
            raw_jd_summary = generated[:500]
        object.__setattr__(validated_extraction, "raw_jd_summary", truncate_to_prompt_cap(raw_jd_summary))

        return validated_extraction

    try:
        return _attempt(temp=0)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "First JD extraction parse failed, attempting repair retry",
            error=str(e),
        )
        try:
            return _attempt(temp=0.3)
        except (json.JSONDecodeError, ValueError) as e2:
            logger.error(
                "Repair retry also failed for JD extraction",
                error=str(e2),
            )
            raise LLMUnavailableError(f"LLM returned malformed JD extraction response: {e2}")



async def _run_ingestion_with_retry(
    entity_type: str,
    entity_id: int,
    event_id: str,
    callback_url: str,
    ingest_fn: Callable[[], Awaitable[None]],
    store: IngestionStatusStore,
    callback_client: CallbackClient,
    ingest_queue: Optional[IngestQueue],
    settings,
    suppress_callback: bool,
) -> None:
    _suppress_callback = suppress_callback
    store.update(event_id, status="running")

    error_summary = None
    max_retries = 3
    total_attempts = max_retries + 1
    backoff_base = 2

    for attempt in range(total_attempts):
        try:
            await ingest_fn()
            error_summary = None
            store.update(event_id, status="success", attempt_count=attempt + 1)
            logger.info("Ingestion succeeded", event_id=event_id, entity_type=entity_type, entity_id=entity_id)
            break

        except Exception as e:
            error_summary = f"{type(e).__name__}: {str(e)}"

            if isinstance(e, LLMUnavailableError) and "circuit breaker is open" in str(e).lower():
                store.update(event_id, status="failed", error_summary=error_summary[:500])
                logger.error("Aborting retries — circuit breaker is open", event_id=event_id)
                break

            logger.warning(
                "Ingestion attempt failed",
                event_id=event_id,
                entity_type=entity_type,
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
                if ingest_queue is not None:
                    record = store.get_by_event_id(event_id)
                    ingest_queue.enqueue(record, backoff_base=settings.INGEST_QUEUE_BACKOFF_BASE_SECONDS)
                    _suppress_callback = True
                logger.error("Ingestion failed after all retries", event_id=event_id, entity_type=entity_type)
            else:
                backoff = backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff)
                continue

    if not _suppress_callback:
        try:
            delivered = await callback_client.send(
                callback_url=callback_url,
                event_id=event_id,
                entity_type=entity_type,
                entity_id=entity_id,
                status="success" if error_summary is None else "failed",
                error=error_summary,
            )
            if not delivered:
                store.update(event_id, callback_delivery_failed=True)
                logger.warning("Callback delivery failed", event_id=event_id)
        except Exception as e:
            logger.error("Callback dispatch failed unexpectedly", event_id=event_id, error=str(e))
            store.update(event_id, callback_delivery_failed=True)


async def run_candidate_ingestion(
    candidate_id: int,
    cv_url: str,
    profile_data: dict,
    callback_url: str,
    event_id: str,
    qdrant: QdrantClient,
    llm: LLMClient,
    store: IngestionStatusStore,
    callback_client: CallbackClient,
    ingest_queue: Optional[IngestQueue] = None,
    suppress_callback: bool = False,
    correlation_id: Optional[str] = None,
) -> None:
    settings = get_settings()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    logger.info("Started candidate ingestion", event_id=event_id, candidate_id=candidate_id)

    validated_profile = ProfileData.model_validate(profile_data)

    async def _ingest() -> None:
        cv_text = await asyncio.to_thread(fetch_and_parse_cv, cv_url)
        new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        cv_text = truncate_to_prompt_cap(cv_text)
        existing_payload = qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if existing_payload is not None and existing_payload is not MISSING and existing_payload.get("cv_hash") == new_hash:
            qdrant.update_payload(CANDIDATES_COLLECTION, candidate_id, {
                "name": validated_profile.name,
                "location": validated_profile.location,
                "experience_level": validated_profile.experience_level,
                "industry": validated_profile.industry,
                "employment_type": validated_profile.employment_type,
                "candidate_version": validated_profile.candidate_version,
                "data_source": validated_profile.data_source,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "cv_hash": new_hash,
                "total_years_experience": existing_payload.get("total_years_experience"),
            })
            logger.info("Candidate ingestion skipped (hash match)", event_id=event_id, candidate_id=candidate_id)
            store.update(event_id, status="success", attempt_count=1)
            return
        logger.debug("CV parsed", event_id=event_id, text_length=len(cv_text))

        profile_data_json = validated_profile.model_dump_json(indent=2)
        validated_extraction = await asyncio.to_thread(
            extract_candidate_entities, cv_text, profile_data_json, llm,
        )

        raw_profile_summary = validated_extraction.raw_profile_summary
        if not raw_profile_summary:
            raw_profile_summary = validated_extraction.raw_profile_summary or ""
        raw_profile_summary = truncate_to_prompt_cap(raw_profile_summary)

        vector = await asyncio.to_thread(llm.embed, raw_profile_summary)

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
            "data_source": validated_profile.data_source,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "cv_hash": new_hash,
            "total_years_experience": validated_extraction.total_years_experience,
        }

        qdrant.upsert(CANDIDATES_COLLECTION, candidate_id, vector, payload)
        logger.info("Candidate ingestion succeeded", event_id=event_id, candidate_id=candidate_id)

    await _run_ingestion_with_retry(
        entity_type="candidate",
        entity_id=candidate_id,
        event_id=event_id,
        callback_url=callback_url,
        ingest_fn=_ingest,
        store=store,
        callback_client=callback_client,
        ingest_queue=ingest_queue,
        settings=settings,
        suppress_callback=suppress_callback,
    )


async def run_job_ingestion(
    job_id: int,
    jd_text: str,
    metadata: dict,
    callback_url: str,
    event_id: str,
    qdrant: QdrantClient,
    llm: LLMClient,
    store: IngestionStatusStore,
    callback_client: CallbackClient,
    ingest_queue: Optional[IngestQueue] = None,
    suppress_callback: bool = False,
    correlation_id: Optional[str] = None,
) -> None:
    settings = get_settings()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    logger.info("Started job ingestion", event_id=event_id, job_id=job_id)

    validated_metadata = JobMetadata.model_validate(metadata)
    new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
    jd_text = truncate_to_prompt_cap(jd_text)

    async def _ingest() -> None:
        existing_payload = qdrant.get(JOBS_COLLECTION, job_id)
        if existing_payload is not None and existing_payload is not MISSING and existing_payload.get("jd_hash") == new_hash:
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
            store.update(event_id, status="success", attempt_count=1)
            return
        metadata_json = validated_metadata.model_dump_json(indent=2)
        validated_extraction = await asyncio.to_thread(
            extract_job_entities, jd_text, metadata_json, llm,
        )

        raw_jd_summary = validated_extraction.raw_jd_summary
        if not raw_jd_summary:
            raw_jd_summary = validated_extraction.raw_jd_summary or ""
        raw_jd_summary = truncate_to_prompt_cap(raw_jd_summary)

        vector = await asyncio.to_thread(llm.embed, raw_jd_summary)

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
        logger.info("Job ingestion succeeded", event_id=event_id, job_id=job_id)

    await _run_ingestion_with_retry(
        entity_type="job",
        entity_id=job_id,
        event_id=event_id,
        callback_url=callback_url,
        ingest_fn=_ingest,
        store=store,
        callback_client=callback_client,
        ingest_queue=ingest_queue,
        settings=settings,
        suppress_callback=suppress_callback,
    )
