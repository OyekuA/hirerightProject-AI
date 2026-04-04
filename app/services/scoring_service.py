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
from app.utils import truncate_to_prompt_cap, parse_gemini_json

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
                if not candidate_payload:
                    logger.warning(
                        "Candidate not found in vector store (stale cache)",
                        candidate_id=candidate_id,
                        collection=CANDIDATES_COLLECTION,
                    )
                    self.cache.delete(cache_key)
                    raise ValueError("Candidate not found")
                job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
                if not job_payload:
                    logger.warning(
                        "Job not found in vector store (stale cache)",
                        job_id=job_id,
                        collection=JOBS_COLLECTION,
                    )
                    self.cache.delete(cache_key)
                    raise ValueError("Job not found")
                candidate_version_in_store = candidate_payload.get("candidate_version", 1)
                job_version_in_store = job_payload.get("job_version", 1)
                if candidate_version_in_store != candidate_version or job_version_in_store != job_version:
                    logger.warning(
                        "Version mismatch detected (stale cache)",
                        candidate_id=candidate_id,
                        expected_candidate_version=candidate_version,
                        found_candidate_version=candidate_version_in_store,
                        job_id=job_id,
                        expected_job_version=job_version,
                        found_job_version=job_version_in_store,
                    )
                    self.cache.delete(cache_key)
                else:
                    return cached

        logger.info("Cache miss or forced refresh", cache_key=cache_key)

        candidate_payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if not candidate_payload:
            logger.warning(
                "Candidate not found in vector store",
                candidate_id=candidate_id,
                collection=CANDIDATES_COLLECTION,
            )
            raise ValueError("Candidate not found")

        job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
        if not job_payload:
            logger.warning(
                "Job not found in vector store",
                job_id=job_id,
                collection=JOBS_COLLECTION,
            )
            raise ValueError("Job not found")

        prompt = f"""
You are a senior recruiter evaluating a candidate for a specific job opening.

CANDIDATE PROFILE (ID {candidate_id}, version {candidate_version}):
```json
{json.dumps(candidate_payload, indent=2)}
```

JOB PROFILE (ID {job_id}, version {job_version}):
```json
{json.dumps(job_payload, indent=2)}
```

Your task is to evaluate the candidate's fit for this job and produce a JSON object with exactly the following structure:

{{
  "overall_score_percentage": <integer between 0 and 100>,
  "category_breakdown": {{
    "role_match":      {{"status": "pass|warning|fail", "short_reason": "..."}},
    "experience":      {{"status": "pass|warning|fail", "short_reason": "..."}},
    "location":        {{"status": "pass|warning|fail", "short_reason": "..."}},
    "employment_type": {{"status": "pass|warning|fail", "short_reason": "..."}}
  }},
  "skill_gap_analysis": "A concise paragraph describing the most significant skill gaps and how the candidate could bridge them."
}}

Rules:
- Return **only** the JSON object, no markdown fences, no extra text.
- For each category, choose "pass", "warning", or "fail" based on your professional judgment.
- Provide a short_reason (1‑2 sentences) explaining the status.
- The overall_score_percentage should reflect the composite suitability (0‑100).
- The skill_gap_analysis must be a plain‑text paragraph.

Now produce the JSON object.
"""

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

        settings = get_settings()
        self.cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(
            "Fit score computed and cached",
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=result["overall_score_percentage"],
        )

        return result