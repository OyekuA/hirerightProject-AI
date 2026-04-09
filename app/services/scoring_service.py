"""Scoring service for calculating fit scores between candidates and jobs.

This module provides the ScoringService class that orchestrates Gemini calls,
Qdrant lookups, and caching to produce a detailed fit score.
"""

import json
import structlog

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.config import get_settings
from app.utils.ingestion import truncate_to_prompt_cap
from app.utils import parse_gemini_json
from app.prompts import SCORING_FIT_PROMPT_TEMPLATE

logger = structlog.get_logger()


class ScoringService:
    """Service that encapsulates Gemini‑based fit‑score calculation."""

    def __init__(
        self,
        gemini: GeminiClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        """Initialize the scoring service.

        Args:
            gemini: A configured GeminiClient instance.
            qdrant: A QdrantClient instance.
            cache: A CacheBackend instance.
        """
        self.gemini = gemini
        self.qdrant = qdrant
        self.cache = cache

    def calculate_fit(
        self,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
        force_refresh: bool = False,
    ) -> dict:
        """Calculate a detailed fit score between a candidate and a job.

        Args:
            candidate_id: Unique identifier of the candidate in the vector store.
            candidate_version: Version of the candidate profile.
            job_id: Unique identifier of the job in the vector store.
            job_version: Version of the job profile.
            force_refresh: If True, bypass the cache and recompute the score.

        Returns:
            A dict with the following keys:
                - overall_score_percentage (int)
                - category_breakdown (dict with four sub‑keys)
                - skill_gap_analysis (str)

        Raises:
            ValueError: If the candidate or job is not found in the vector store.
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails,
                or if the response is malformed.
        """
        cache_key = f"{candidate_id}:{candidate_version}:{job_id}:{job_version}"
        logger.info(
            "Calculating fit score",
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            job_id=job_id,
            job_version=job_version,
            force_refresh=force_refresh,
            cache_key=cache_key,
        )

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit", cache_key=cache_key)
                candidate_payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
                job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
                if candidate_payload and job_payload:
                    cached_candidate_version = candidate_payload.get("candidate_version", 1)
                    cached_job_version = job_payload.get("job_version", 1)
                    if cached_candidate_version == candidate_version and cached_job_version == job_version:
                        return cached
                    else:
                        logger.warning(
                            "Stale cache entry due to version mismatch, deleting",
                            cache_key=cache_key,
                            cached_candidate_version=cached_candidate_version,
                            expected_candidate_version=candidate_version,
                            cached_job_version=cached_job_version,
                            expected_job_version=job_version,
                        )
                        self.cache.delete(cache_key)
                else:
                    logger.warning(
                        "Stale cache entry for deleted entity, deleting",
                        cache_key=cache_key,
                        candidate_exists=bool(candidate_payload),
                        job_exists=bool(job_payload),
                    )
                    self.cache.delete(cache_key)

        logger.info("Cache miss or forced refresh", cache_key=cache_key)

        candidate_payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if not candidate_payload:
            logger.warning(
                "Candidate not found in vector store",
                candidate_id=candidate_id,
                collection=CANDIDATES_COLLECTION,
            )
            raise ValueError("Candidate not found")
        stored_candidate_version = candidate_payload.get("candidate_version", 1)
        if stored_candidate_version != candidate_version:
            logger.warning(
                "Candidate version mismatch",
                candidate_id=candidate_id,
                stored_version=stored_candidate_version,
                requested_version=candidate_version,
            )
            raise ValueError(f"Candidate version mismatch: stored {stored_candidate_version}, requested {candidate_version}")

        job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
        if not job_payload:
            logger.warning(
                "Job not found in vector store",
                job_id=job_id,
                collection=JOBS_COLLECTION,
            )
            raise ValueError("Job not found")
        stored_job_version = job_payload.get("job_version", 1)
        if stored_job_version != job_version:
            logger.warning(
                "Job version mismatch",
                job_id=job_id,
                stored_version=stored_job_version,
                requested_version=job_version,
            )
            raise ValueError(f"Job version mismatch: stored {stored_job_version}, requested {job_version}")

        candidate_payload_json = json.dumps(candidate_payload, indent=2)
        job_payload_json = json.dumps(job_payload, indent=2)
        prompt = SCORING_FIT_PROMPT_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            candidate_payload_json=candidate_payload_json,
            job_id=job_id,
            job_version=job_version,
            job_payload_json=job_payload_json
        )

        prompt = truncate_to_prompt_cap(prompt)
        generated = self.gemini.generate(prompt)

        try:
            result = parse_gemini_json(generated)
        except json.JSONDecodeError as e:
            logger.error(
                "Gemini returned non‑JSON response",
                error=str(e),
                raw=generated[:500],
            )
            raise GeminiUnavailableError(
                f"Gemini returned malformed fit-score JSON: {e}"
            )

        if not isinstance(result, dict):
            logger.error(
                "Gemini returned a non‑dict JSON payload",
                payload_type=type(result).__name__,
                raw=generated[:500],
            )
            raise GeminiUnavailableError(
                "Gemini returned malformed fit‑score JSON: expected a dict"
            )

        required_top = {"overall_score_percentage", "category_breakdown", "skill_gap_analysis"}
        if not all(k in result for k in required_top):
            logger.error(
                "Gemini response missing required top‑level keys",
                response_keys=list(result.keys()),
                required_keys=list(required_top),
            )
            raise GeminiUnavailableError(
                "Gemini response missing required top‑level keys"
            )

        category_keys = {"role_match", "experience", "location", "employment_type"}
        cat_breakdown = result.get("category_breakdown")
        if not isinstance(cat_breakdown, dict) or not all(
            k in cat_breakdown for k in category_keys
        ):
            logger.error(
                "Gemini category_breakdown missing or malformed",
                category_breakdown=cat_breakdown,
            )
            raise GeminiUnavailableError(
                "Gemini category_breakdown missing required sub‑keys"
            )

        for key in category_keys:
            sub = cat_breakdown[key]
            if not isinstance(sub, dict) or "status" not in sub or "short_reason" not in sub:
                logger.error(
                    f"Category {key} is missing 'status' or 'short_reason'",
                    sub=sub,
                )
                raise GeminiUnavailableError(
                    f"Category {key} is missing required fields"
                )
            if sub["status"] not in ("pass", "warning", "fail"):
                logger.error(
                    f"Unexpected status value in category {key}",
                    status=sub["status"],
                )
                raise GeminiUnavailableError(
                    f"Gemini returned invalid status '{sub['status']}' in category {key}"
                )

        try:
            overall = int(result["overall_score_percentage"])
        except (TypeError, ValueError):
            logger.error(
                "Gemini returned non-integer overall_score_percentage",
                value=result["overall_score_percentage"],
            )
            raise GeminiUnavailableError(
                "overall_score_percentage must be an integer between 0 and 100"
            )
        if not (0 <= overall <= 100):
            logger.error(
                "Gemini returned out-of-range overall_score_percentage",
                value=overall,
            )
            raise GeminiUnavailableError(
                "overall_score_percentage must be an integer between 0 and 100"
            )
        result["overall_score_percentage"] = overall

        settings = get_settings()
        self.cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(
            "Fit score computed and cached",
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=result["overall_score_percentage"],
        )

        return result