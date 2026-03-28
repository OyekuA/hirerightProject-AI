"""Career path analysis service.

This module provides the CareerPathService class that orchestrates Gemini calls
and Qdrant lookups to suggest three suitable career paths for a candidate.
"""

import json
import structlog

from app.clients.dependencies import CANDIDATES_COLLECTION
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils import truncate_to_prompt_cap

logger = structlog.get_logger()


class CareerPathService:
    """Service that encapsulates Gemini‑based career path analysis."""

    def __init__(self, gemini: GeminiClient, qdrant: QdrantClient):
        """Initialize the career path service.

        Args:
            gemini: A configured GeminiClient instance.
            qdrant: A QdrantClient instance.
        """
        self.gemini = gemini
        self.qdrant = qdrant

    def analyze_career_paths(self, candidate_id: int) -> list[dict]:
        """Suggest three career paths based on the candidate's profile.

        Args:
            candidate_id: Unique identifier of the candidate in the vector store.

        Returns:
            A list of exactly three dicts, each with keys:
                - role (string)
                - match_percentage (integer 0‑100)
                - reasoning (string, one sentence)

        Raises:
            ValueError: If the candidate is not found in the vector store.
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails,
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

        prompt = f"""
You are a senior career advisor evaluating a candidate's profile.

CANDIDATE PROFILE (ID {candidate_id}):
```json
{json.dumps(payload, indent=2)}
```

Your task is to suggest exactly three career paths that would be a good fit for this candidate.
For each path, produce a JSON object with the following keys:

- "role": a string describing the job title or role (e.g., "Senior Data Engineer")
- "match_percentage": an integer between 0 and 100 indicating how well the candidate's profile matches this role
- "reasoning": a single concise sentence explaining why this role is a good fit

Return **only** a JSON array containing exactly three objects, no markdown fences, no extra text.
The array must be formatted as follows:

[
  {{"role": "...", "match_percentage": ..., "reasoning": "..."}},
  {{"role": "...", "match_percentage": ..., "reasoning": "..."}},
  {{"role": "...", "match_percentage": ..., "reasoning": "..."}}
]

Now produce the JSON array.
"""

        prompt = truncate_to_prompt_cap(prompt)
        generated = self.gemini.generate(prompt)

        try:
            result = json.loads(generated.strip())
        except json.JSONDecodeError as e:
            logger.error(
                "Gemini returned non‑JSON response",
                error=str(e),
            )
            raise GeminiUnavailableError(
                f"Gemini returned malformed career‑paths JSON: {e}"
            )

        if not isinstance(result, list):
            logger.error(
                "Gemini response is not a list",
                response_type=type(result),
            )
            raise GeminiUnavailableError("Gemini response is not a list")

        if len(result) != 3:
            logger.error(
                "Gemini response does not contain exactly three items",
                num_items=len(result),
            )
            raise GeminiUnavailableError(
                f"Gemini returned {len(result)} items, expected 3"
            )

        required_keys = {"role", "match_percentage", "reasoning"}
        for i, item in enumerate(result):
            if not isinstance(item, dict):
                logger.error(
                    f"Item {i} is not a dict",
                    item=item,
                )
                raise GeminiUnavailableError(f"Item {i} is not a dict")
            missing = required_keys - set(item.keys())
            if missing:
                logger.error(
                    f"Item {i} missing required keys",
                    missing_keys=list(missing),
                    item_keys=list(item.keys()),
                )
                raise GeminiUnavailableError(
                    f"Item {i} missing required keys: {missing}"
                )
            perc = item.get("match_percentage")
            if not isinstance(perc, int) or perc < 0 or perc > 100:
                logger.error(
                    f"Item {i} has invalid match_percentage",
                    match_percentage=perc,
                )
                raise GeminiUnavailableError(
                    f"Item {i} match_percentage must be an integer 0‑100"
                )
            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                logger.error(
                    f"Item {i} has invalid role",
                    role=role,
                )
                raise GeminiUnavailableError(
                    f"Item {i} role must be a non‑empty string"
                )
            reasoning = item.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                logger.error(
                    f"Item {i} has invalid reasoning",
                    reasoning=reasoning,
                )
                raise GeminiUnavailableError(
                    f"Item {i} reasoning must be a non‑empty string"
                )

        logger.info(
            "Career paths analyzed successfully",
            candidate_id=candidate_id,
            roles=[item["role"] for item in result],
        )
        return result