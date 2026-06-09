import json
import structlog

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.config import get_settings
from app.utils.ingestion import truncate_to_prompt_cap
from app.utils import parse_llm_json
from app.prompts import SCORING_FIT_PROMPT_TEMPLATE

logger = structlog.get_logger()


class ScoringService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache

    def _run_scoring(self, prompt: str) -> dict:
        generated = self.llm.generate(prompt, temperature=0)
        result = parse_llm_json(generated)

        if not isinstance(result, dict):
            raise LLMUnavailableError("LLM returned a non‑dict payload")

        required_top = {"overall_score_percentage", "category_breakdown", "skill_gap_analysis"}
        if not all(k in result for k in required_top):
            raise LLMUnavailableError("LLM response missing required top‑level keys")

        category_keys = {"role_match", "experience", "location", "employment_type"}
        cat_breakdown = result.get("category_breakdown")
        if not isinstance(cat_breakdown, dict) or not all(
            k in cat_breakdown for k in category_keys
        ):
            raise LLMUnavailableError("category_breakdown missing required sub‑keys")

        for key in category_keys:
            sub = cat_breakdown[key]
            if not isinstance(sub, dict) or "status" not in sub or "short_reason" not in sub:
                raise LLMUnavailableError(f"Category {key} is missing required fields")
            if sub["status"] not in ("pass", "warning", "fail"):
                raise LLMUnavailableError(f"Invalid status '{sub['status']}' in category {key}")

        try:
            overall = int(result["overall_score_percentage"])
        except (TypeError, ValueError):
            raise LLMUnavailableError("overall_score_percentage must be an integer")
        if not (0 <= overall <= 100):
            raise LLMUnavailableError("overall_score_percentage out of range 0‑100")
        result["overall_score_percentage"] = overall

        return result

    def calculate_fit(
        self,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
        force_refresh: bool = False,
    ) -> dict:
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
        result = self._run_scoring(prompt)

        settings = get_settings()
        self.cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(
            "Fit score computed and cached",
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=result["overall_score_percentage"],
        )

        return result
