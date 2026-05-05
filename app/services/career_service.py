"""Career path analysis service.

This module provides the CareerPathService class that orchestrates LLM calls
and Qdrant lookups to suggest three suitable career paths for a candidate.
"""

import json
import structlog

from app.clients.dependencies import CANDIDATES_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils import parse_llm_json
from app.utils.ingestion import truncate_to_prompt_cap
from app.prompts import CAREER_PATHS_PROMPT_TEMPLATE
from app.schemas.career import AnalyzeCareerPathsResponse

logger = structlog.get_logger()


class CareerPathService:
    """Service that encapsulates LLM‑based career path analysis."""

    def __init__(self, llm: LLMClient, qdrant: QdrantClient):
        """Initialize the career path service.

        Args:
            llm: A configured LLMClient instance.
            qdrant: A QdrantClient instance.
        """
        self.llm = llm
        self.qdrant = qdrant

    def analyze_career_paths(self, candidate_id: int) -> dict:
        """Suggest three career paths based on the candidate's profile.

        Args:
            candidate_id: Unique identifier of the candidate in the vector store.

        Returns:
            A dict with keys:
                - profile_summary (string): a 2‑3 sentence second‑person summary of the
                  candidate's overall profile.
                - paths (list): a list of exactly three dicts, each with keys:
                    - role (string)
                    - match_percentage (integer 0‑100)
                    - core_skills (list of strings, 3‑5 skills)
                    - reasoning (string, one sentence in second‑person voice)

        Raises:
            ValueError: If the candidate is not found in the vector store.
            LLMUnavailableError: If the LLM circuit breaker is open or the call fails,
                or if the response is malformed.
        """
        logger.info(
            "Analyzing career paths",
            candidate_id=candidate_id,
        )

        payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if payload is None or payload == {}:
            logger.warning(
                "Candidate not found in vector store",
                candidate_id=candidate_id,
                collection=CANDIDATES_COLLECTION,
            )
            raise ValueError("Candidate not found in vector store")

        candidate_payload_json = json.dumps(payload, indent=2)
        prompt = CAREER_PATHS_PROMPT_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_payload_json=candidate_payload_json
        )

        prompt = truncate_to_prompt_cap(prompt)
        generated = self.llm.generate(prompt)

        try:
            result = parse_llm_json(generated)
        except json.JSONDecodeError as e:
            logger.error(
                "LLM returned non‑JSON response",
                error=str(e),
            )
            raise LLMUnavailableError(
                f"LLM returned malformed career‑paths JSON: {e}"
            )

        if not isinstance(result, dict):
            logger.error(
                "LLM response is not a dict",
                response_type=type(result),
            )
            raise LLMUnavailableError("LLM response is not a dict")

        if "profile_summary" not in result or not isinstance(result["profile_summary"], str) or not result["profile_summary"].strip():
            logger.error(
                "LLM response missing or invalid top-level profile_summary",
                profile_summary=result.get("profile_summary"),
            )
            raise LLMUnavailableError("LLM response missing or invalid top-level profile_summary")

        if "paths" not in result or not isinstance(result["paths"], list):
            logger.error(
                "LLM response missing or invalid paths",
                paths=result.get("paths"),
            )
            raise LLMUnavailableError("LLM response missing or invalid paths")

        if len(result["paths"]) != 3:
            logger.error(
                "LLM response does not contain exactly three items in paths",
                num_items=len(result["paths"]),
            )
            raise LLMUnavailableError(
                f"LLM returned {len(result['paths'])} items in paths, expected 3"
            )

        required_keys = {"role", "match_percentage", "reasoning", "core_skills"}
        for i, item in enumerate(result["paths"]):
            if not isinstance(item, dict):
                logger.error(
                    f"Item {i} is not a dict",
                    item=item,
                )
                raise LLMUnavailableError(f"Item {i} is not a dict")
            missing = required_keys - set(item.keys())
            if missing:
                logger.error(
                    f"Item {i} missing required keys",
                    missing_keys=list(missing),
                    item_keys=list(item.keys()),
                )
                raise LLMUnavailableError(
                    f"Item {i} missing required keys: {missing}"
                )
            perc = item.get("match_percentage")
            if not isinstance(perc, int) or perc < 0 or perc > 100:
                logger.error(
                    f"Item {i} has invalid match_percentage",
                    match_percentage=perc,
                )
                raise LLMUnavailableError(
                    f"Item {i} match_percentage must be an integer 0‑100"
                )
            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                logger.error(
                    f"Item {i} has invalid role",
                    role=role,
                )
                raise LLMUnavailableError(
                    f"Item {i} role must be a non‑empty string"
                )
            reasoning = item.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                logger.error(
                    f"Item {i} has invalid reasoning",
                    reasoning=reasoning,
                )
                raise LLMUnavailableError(
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

        logger.info(
            "Career paths analyzed successfully",
            candidate_id=candidate_id,
            roles=[item["role"] for item in result["paths"]],
            core_skills_counts=[len(item.get("core_skills", [])) for item in result["paths"]],
        )
        return AnalyzeCareerPathsResponse.model_validate(result).model_dump(mode="json")
