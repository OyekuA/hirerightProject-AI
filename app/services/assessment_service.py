"""Assessment service for generating interview questions and grading candidate answers.

This module provides the AssessmentService class that orchestrates Gemini calls
and Qdrant lookups to produce scenario‑based questions and evaluate answers.
"""

import json
import structlog
from typing import Optional

from app.clients.dependencies import CANDIDATES_COLLECTION
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils import truncate_to_prompt_cap, parse_gemini_json

logger = structlog.get_logger()


class AssessmentService:
    """Service that encapsulates Gemini‑based assessment generation and grading."""

    def __init__(self, gemini: GeminiClient, qdrant: Optional[QdrantClient] = None):
        """Initialize the assessment service.

        Args:
            gemini: A configured GeminiClient instance.
            qdrant: A QdrantClient instance (optional, required only for generate_questions).
        """
        self.gemini = gemini
        self.qdrant = qdrant

    def generate_questions(
        self, candidate_id: int, target_role: str, num_questions: int
    ) -> list[str]:
        """Generate scenario‑based interview questions tailored to a candidate.

        Args:
            candidate_id: Unique identifier of the candidate in the vector store.
            target_role: The role the candidate is being assessed for.
            num_questions: Desired number of questions (clamped to [1, 5]).

        Returns:
            A list of exactly N scenario‑based questions.

        Raises:
            ValueError: If the candidate is not found in the vector store.
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
        """
        clamped_n = max(1, min(5, num_questions))
        logger.info(
            "Generating assessment questions",
            candidate_id=candidate_id,
            target_role=target_role,
            requested_questions=num_questions,
            clamped_questions=clamped_n,
        )

        if self.qdrant is None:
            raise ValueError("Qdrant client is required for question generation")
        payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if payload is None:
            logger.warning("Candidate not found in vector store", candidate_id=candidate_id)
            raise ValueError("Candidate not found in vector store")

        past_roles = payload.get("past_roles", [])
        skills = payload.get("skills", [])

        def normalize_role(role):
            if isinstance(role, str):
                return role
            if isinstance(role, dict):
                parts = []
                if "company" in role:
                    parts.append(role["company"])
                if "title" in role:
                    parts.append(role["title"])
                if "duration" in role:
                    parts.append(role["duration"])
                return " – ".join(parts) if parts else str(role)
            return str(role)

        normalized_past_roles = [normalize_role(r) for r in past_roles]

        prompt = f"""
You are an expert interviewer for a {target_role} position.

The candidate has the following background:
- Previous roles: {', '.join(normalized_past_roles) if normalized_past_roles else 'No previous roles listed'}
- Skills: {', '.join(skills) if skills else 'No skills listed'}

Generate exactly {clamped_n} scenario‑based interview questions that:
1. Are tailored to the candidate’s specific past experience and skills.
2. Pose concrete, realistic work situations they might encounter in the {target_role} role.
3. Require the candidate to explain how they would apply their past experience to solve the scenario.
4. Forbid generic definition questions (e.g., “What is X?”) or trivia.

Return a JSON array of strings, where each string is one question, and nothing else.

Example format:
[
  "Describe a time when you had to ...",
  "Imagine you are faced with ..."
]
"""
        generated = self.gemini.generate(prompt)
        try:
            parsed = parse_gemini_json(generated)
            if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
                raise ValueError("Gemini response is not a list of strings")
            questions = parsed
            if len(questions) > clamped_n:
                logger.warning(
                    "Gemini returned more questions than requested, truncating",
                    requested=clamped_n,
                    received=len(questions),
                )
                questions = questions[:clamped_n]
            elif len(questions) < clamped_n:
                logger.error(
                    "Gemini returned fewer questions than requested",
                    requested=clamped_n,
                    received=len(questions),
                )
                raise GeminiUnavailableError(
                    f"Gemini returned only {len(questions)} questions, expected {clamped_n}"
                )
            return questions
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Gemini response as JSON array", error=str(e))
            raise GeminiUnavailableError(f"Gemini returned malformed response: {e}")

    def grade_answers(
        self, questions: list[str], answers: list[str], time_taken_seconds: int
    ) -> dict:
        """Grade a candidate's answers and produce a score, feedback, and authenticity flag.

        Args:
            questions: List of questions that were asked.
            answers: List of answers provided by the candidate (same order as questions).
            time_taken_seconds: Total time the candidate spent answering all questions.

        Returns:
            A dictionary with the following keys:
                - overall_score (int 0‑100)
                - feedback (str)
                - authenticity_flag (dict with keys is_suspicious and reason)

        Raises:
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
        """
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")

        logger.info(
            "Grading assessment answers",
            num_questions=len(questions),
            time_taken_seconds=time_taken_seconds,
        )

        truncated_answers = [truncate_to_prompt_cap(ans) for ans in answers]
        qa_pairs = "\n".join(
            f"Question {i+1}: {q}\nAnswer {i+1}: {a}"
            for i, (q, a) in enumerate(zip(questions, truncated_answers))
        )

        prompt = f"""
You are an expert interview grader. Below are the interview questions and the candidate's answers.

The candidate completed all answers in {time_taken_seconds} seconds.

{qa_pairs}

Evaluate the candidate's performance and return a JSON object with exactly these three keys:

1. "overall_score": an integer between 0 and 100 representing the overall quality of the answers.
   Consider relevance, depth, clarity, and alignment with the role's expectations.

2. "feedback": a string containing constructive, actionable feedback for the candidate.
   Highlight strengths and suggest concrete improvements.

3. "authenticity_flag": an object with two keys:
   - "is_suspicious": a boolean indicating whether the answers raise suspicion of being AI‑generated or rushed.
   - "reason": a string explaining why the flag is set (or not). Use the time_taken_seconds as context,
     but do NOT automatically flag fast answers as suspicious—consider content quality and coherence.
     A suspicious flag should be set only if there are strong indicators (e.g., generic, formulaic,
     contradictory, or off‑topic responses).

Return only the JSON object, no other text.

Example format:
{{
  "overall_score": 85,
  "feedback": "The candidate demonstrated solid practical experience...",
  "authenticity_flag": {{ "is_suspicious": false, "reason": "Answers were specific and coherent." }}
}}
"""
        generated = self.gemini.generate(prompt)
        try:
            result = parse_gemini_json(generated)
            required_keys = {"overall_score", "feedback", "authenticity_flag"}
            if not all(k in result for k in required_keys):
                raise ValueError("Missing required keys in Gemini response")
            auth_flag = result["authenticity_flag"]
            if not isinstance(auth_flag, dict) or "is_suspicious" not in auth_flag or "reason" not in auth_flag:
                raise ValueError("authenticity_flag is malformed")
            overall_score = int(result["overall_score"])
            if not (0 <= overall_score <= 100):
                raise ValueError(f"overall_score {overall_score} is out of bounds (0‑100)")
            result["overall_score"] = overall_score
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error("Failed to parse Gemini grading response", error=str(e))
            raise GeminiUnavailableError(f"Gemini returned malformed grading response: {e}")