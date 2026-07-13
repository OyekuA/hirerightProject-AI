import json
import structlog

from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.config import get_settings
from app.utils import parse_llm_json
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import INTERVIEW_RECOMMENDATION_PROMPT_TEMPLATE
from app.services.scoring_service import ScoringService

logger = structlog.get_logger()


class DecisionService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache
        self._scoring_service = ScoringService(llm=llm, qdrant=qdrant, cache=cache)

    @staticmethod
    def _compute_label(combined_score: int, assessment_score: int) -> str:
        if combined_score >= 80 and assessment_score >= 75:
            return "hire"
        if combined_score < 50 or assessment_score < 40:
            return "no_hire"
        return "review"

    @staticmethod
    def _parse_rationale(raw: str) -> dict:
        try:
            result = parse_llm_json(raw)
        except json.JSONDecodeError:
            raise LLMUnavailableError("LLM returned malformed rationale JSON")
        if not isinstance(result, dict):
            raise LLMUnavailableError("LLM returned a non-dict rationale payload")
        missing = {"rationale", "confidence"} - set(result.keys())
        if missing:
            raise LLMUnavailableError(f"LLM rationale missing required keys: {missing}")
        confidence = result.get("confidence", 0)
        try:
            confidence = int(confidence)
        except (TypeError, ValueError):
            raise LLMUnavailableError("LLM 'confidence' must be an integer")
        if not (0 <= confidence <= 100):
            raise LLMUnavailableError("LLM 'confidence' must be 0-100")
        return result

    def decide(
        self,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
        assessment_score: int,
    ) -> dict:
        logger.info(
            "Running decision engine",
            candidate_id=candidate_id,
            job_id=job_id,
            assessment_score=assessment_score,
        )

        fit_result = self._scoring_service.calculate_fit(
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            job_id=job_id,
            job_version=job_version,
        )

        fit_score = fit_result["overall_score_percentage"]
        settings = get_settings()
        combined_score = round(
            settings.DECISION_FIT_WEIGHT * fit_score
            + settings.DECISION_ASSESSMENT_WEIGHT * assessment_score
        )
        combined_score = max(0, min(100, combined_score))

        decision = self._compute_label(combined_score, assessment_score)

        prompt = INTERVIEW_RECOMMENDATION_PROMPT_TEMPLATE.format(
            decision=decision,
            combined_score=combined_score,
            assessment_score=assessment_score,
            category_breakdown_json=json.dumps(fit_result["category_breakdown"], indent=2),
        )
        prompt = truncate_to_prompt_cap(prompt)

        raw = self.llm.generate(prompt, temperature=0)
        rationale = self._parse_rationale(raw)

        return {
            "decision": decision,
            "combined_score": combined_score,
            "fit_score": fit_score,
            "assessment_score": assessment_score,
            "rationale": rationale["rationale"],
            "confidence": rationale["confidence"],
        }
