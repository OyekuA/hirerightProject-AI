import json
import structlog
from typing import Optional

from app.clients.cache import CacheBackend
from app.clients.dependencies import CANDIDATES_COLLECTION
from app.clients.llm import LLMClient
from app.clients.qdrant import QdrantClient, MISSING
from app.config import get_settings
from app.utils import parse_llm_json
from app.utils.bias_masking import mask_candidate_for_scoring
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import CAREER_PATHS_PROMPT_TEMPLATE

logger = structlog.get_logger()


class MalformedLLMResponseError(Exception):
    pass


class CareerPathService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: Optional[CacheBackend] = None,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache

    def analyze_career_paths(self, candidate_id: int) -> dict:
        logger.info(
            "Analyzing career paths",
            candidate_id=candidate_id,
        )

        payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if payload is MISSING or payload is None:
            logger.warning(
                "Candidate not found in vector store",
                candidate_id=candidate_id,
                collection=CANDIDATES_COLLECTION,
            )
            raise ValueError("Candidate not found in vector store")

        candidate_version = payload.get("candidate_version", 0)

        if self.cache is not None:
            cache_key = f"{candidate_id}:{candidate_version}:career"
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "Returning cached career paths",
                    candidate_id=candidate_id,
                )
                return cached

        masked_payload = mask_candidate_for_scoring(payload)
        candidate_payload_json = json.dumps(masked_payload, indent=2)
        prompt = CAREER_PATHS_PROMPT_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_payload_json=candidate_payload_json
        )

        prompt = truncate_to_prompt_cap(prompt)

        settings = get_settings()
        seed = settings.LLM_SEED
        generated = self.llm.generate(
            prompt,
            temperature=0.2,
            response_format={"type": "json_object"},
            seed=seed,
        )

        try:
            result = parse_llm_json(generated)
        except json.JSONDecodeError as e:
            logger.warning(
                "First career paths parse failed, attempting repair retry",
                error=str(e),
            )
            try:
                generated = self.llm.generate(
                    prompt,
                    temperature=0,
                    response_format={"type": "json_object"},
                    seed=seed,
                )
                result = parse_llm_json(generated)
            except json.JSONDecodeError as e2:
                logger.error(
                    "LLM returned non‑JSON response after retry",
                    error=str(e2),
                )
                raise MalformedLLMResponseError(
                    f"LLM returned malformed career‑paths JSON: {e2}"
                )

        if not isinstance(result, dict):
            logger.error(
                "LLM response is not a dict",
                response_type=type(result),
            )
            raise MalformedLLMResponseError("LLM response is not a dict")

        if "profile_summary" not in result or not isinstance(result["profile_summary"], str) or not result["profile_summary"].strip():
            logger.error(
                "LLM response missing or invalid top-level profile_summary",
                profile_summary=result.get("profile_summary"),
            )
            raise MalformedLLMResponseError("LLM response missing or invalid top-level profile_summary")

        if "paths" not in result or not isinstance(result["paths"], list):
            logger.error(
                "LLM response missing or invalid paths",
                paths=result.get("paths"),
            )
            raise MalformedLLMResponseError("LLM response missing or invalid paths")

        if len(result["paths"]) != 3:
            logger.error(
                "LLM response does not contain exactly three items in paths",
                num_items=len(result["paths"]),
            )
            raise MalformedLLMResponseError(
                f"LLM returned {len(result['paths'])} items in paths, expected 3"
            )

        required_keys = {"role", "match_percentage", "reasoning", "core_skills"}
        for i, item in enumerate(result["paths"]):
            if not isinstance(item, dict):
                logger.error(
                    f"Item {i} is not a dict",
                    item=item,
                )
                raise MalformedLLMResponseError(f"Item {i} is not a dict")
            missing = required_keys - set(item.keys())
            if missing:
                logger.error(
                    f"Item {i} missing required keys",
                    missing_keys=list(missing),
                    item_keys=list(item.keys()),
                )
                raise MalformedLLMResponseError(
                    f"Item {i} missing required keys: {missing}"
                )
            perc = item.get("match_percentage")
            if not isinstance(perc, int) or perc < 0 or perc > 100:
                logger.error(
                    f"Item {i} has invalid match_percentage",
                    match_percentage=perc,
                )
                raise MalformedLLMResponseError(
                    f"Item {i} match_percentage must be an integer 0‑100"
                )
            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                logger.error(
                    f"Item {i} has invalid role",
                    role=role,
                )
                raise MalformedLLMResponseError(
                    f"Item {i} role must be a non‑empty string"
                )
            reasoning = item.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                logger.error(
                    f"Item {i} has invalid reasoning",
                    reasoning=reasoning,
                )
                raise MalformedLLMResponseError(
                    f"Item {i} reasoning must be a non‑empty string"
                )
            core_skills = item.get("core_skills")
            if not isinstance(core_skills, list):
                logger.warning(
                    f"Item {i} core_skills is not a list, defaulting to empty",
                    core_skills=core_skills,
                )
                core_skills = []
                item["core_skills"] = core_skills
            valid_skills = [
                s for s in core_skills
                if isinstance(s, str) and s.strip()
            ]
            if len(valid_skills) < len(core_skills):
                logger.warning(
                    f"Item {i} had {len(core_skills) - len(valid_skills)} invalid skill(s) removed",
                    original=core_skills,
                    valid=valid_skills,
                )
            item["core_skills"] = valid_skills

        if self.cache is not None:
            cache_key = f"{candidate_id}:{candidate_version}:career"
            self.cache.set(cache_key, result, ttl=get_settings().CACHE_TTL_SECONDS)

        logger.info(
            "Career paths analyzed successfully",
            candidate_id=candidate_id,
            roles=[item["role"] for item in result["paths"]],
            core_skills_counts=[len(item.get("core_skills", [])) for item in result["paths"]],
        )
        return result
