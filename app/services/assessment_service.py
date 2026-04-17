"""Assessment service for generating interview questions and grading candidate answers.

This module provides the AssessmentService class that orchestrates Gemini calls
and Qdrant lookups to produce scenario‑based questions and evaluate answers.
"""

import json
import random
import structlog
from typing import Optional, Union, Literal, List, Dict, Any

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils import parse_gemini_json
from app.prompts import GENERATE_QUESTIONS_PROMPT_TEMPLATE, GRADE_ANSWERS_PROMPT_TEMPLATE
from ..schemas.assessment import MultipleChoiceQuestion

logger = structlog.get_logger()

HUMAN_TYPING_SPEED_THRESHOLD_WPS = 2.5
AUTHENTICITY_PENALTY = 25


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
        self,
        candidate_id: Optional[int],
        target_role: Optional[str],
        num_questions: int,
        job_id: Optional[int] = None,
        question_type: Literal["single", "multiple_choice"] = "single",
    ) -> List[Union[str, dict]]:
        """Generate scenario‑based interview questions tailored to a candidate or a job."""
        clamped_n = max(1, min(30, num_questions))
        logger.info(
            "Generating assessment questions",
            candidate_id=candidate_id,
            job_id=job_id,
            target_role=target_role,
            requested_questions=num_questions,
            clamped_questions=clamped_n,
            question_type=question_type,
        )

        if candidate_id is None and job_id is None:
            raise ValueError("At least one of candidate_id or job_id must be provided")

        if (candidate_id is not None or job_id is not None) and self.qdrant is None:
            raise ValueError("Qdrant client is required for question generation")

        past_roles = []
        skills = []
        if candidate_id is not None:
            payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
            if payload is None:
                logger.warning("Candidate not found in vector store", candidate_id=candidate_id)
                raise ValueError("Candidate not found in vector store")
            past_roles = payload.get("past_roles", [])
            skills = payload.get("skills", [])

        job_context = {}
        if job_id is not None:
            job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
            if job_payload is None:
                logger.warning("Job not found in vector store", job_id=job_id)
                raise ValueError("Job not found in vector store")
            job_context = {
                "title": job_payload.get("title", ""),
                "required_skills": job_payload.get("required_skills", []),
                "raw_jd_summary": job_payload.get("raw_jd_summary", ""),
            }
            if not target_role:
                target_role = job_context["title"]

        if not target_role:
            raise ValueError(
                "Could not determine target role — provide it explicitly or ensure the job has a title"
            )

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

        candidate_block = ""
        if candidate_id is not None:
            candidate_block = f"""
The candidate has the following background:
- Previous roles: {', '.join(normalized_past_roles) if normalized_past_roles else 'No previous roles listed'}
- Skills: {', '.join(skills) if skills else 'No skills listed'}"""
        else:
            candidate_block = """
The candidate background is not specified (generic assessment)."""

        job_block = ""
        if job_id is not None:
            job_block = f"""
The assessment is for the following role:
- Job Title: {job_context['title']}
- Required Skills: {', '.join(job_context['required_skills']) if job_context['required_skills'] else 'None'}
- Role Summary: {job_context['raw_jd_summary']}"""
        else:
            job_block = """
The job context is not specified (generic role)."""

        if question_type == "single":
            instruction = f"""
Generate exactly {clamped_n} scenario‑based interview questions that:
1. Are tailored to the candidate’s specific past experience and skills (if provided).
2. Pose concrete, realistic work situations they might encounter in the {target_role} role.
3. Require the candidate to explain how they would apply their past experience to solve the scenario.
4. Forbid generic definition questions (e.g., “What is X?”) or trivia.

Return a JSON array of strings, where each string is one question, and nothing else.

Example format:
[
  "Describe a time when you had to ...",
  "Imagine you are faced with ..."
]"""
        else:
            instruction = f"""
Generate exactly {clamped_n} multiple-choice interview questions that:
1. Are tailored to the candidate’s specific past experience and skills.
2. Pose highly complex, tricky, realistic work situations for a {target_role}.
3. The distractors (wrong options) MUST be common industry misconceptions or plausible mistakes.
4. CRITICAL FORMATTING: The options must ONLY contain the proposed action or solution.
   Do NOT include explanations, justifications, or critiques inside the option text itself.
   - BAD OPTION: "Query the monolith database directly. This creates tight coupling..."
   - GOOD OPTION: "Query the monolith's production database directly to retrieve the required data."

Return a JSON array of objects, where each object has the following keys:
- "question": a string containing the scenario-based question.
- "correct_answer": a string with the exact correct answer.
- "distractors": an array of exactly 3 strings containing the plausible wrong answers.
Do NOT prepend A, B, C, D to the answers.
"""

        prompt = GENERATE_QUESTIONS_PROMPT_TEMPLATE.format(
            target_role=target_role,
            job_block=job_block,
            candidate_block=candidate_block,
            instruction=instruction
        )
        generated = self.gemini.generate(prompt)
        try:
            parsed = parse_gemini_json(generated)
            if not isinstance(parsed, list):
                raise ValueError("Gemini response is not a list")
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

            if question_type == "single":
                for q in questions:
                    if not isinstance(q, str):
                        raise ValueError("Gemini response contains non‑string items in single‑question mode")
            else:
                formatted_questions = []
                for i, q in enumerate(questions):
                    if not isinstance(q, dict):
                        raise ValueError(f"Question {i} is not a dict")
                    required_keys = {"question", "correct_answer", "distractors"}
                    missing = required_keys - set(q.keys())
                    if missing:
                        raise ValueError(f"Question {i} missing keys: {missing}")
                    if not isinstance(q["question"], str):
                        raise ValueError(f"Question {i} 'question' must be a string")
                    if not isinstance(q["correct_answer"], str):
                        raise ValueError(f"Question {i} 'correct_answer' must be a string")
                    if not isinstance(q["distractors"], list) or len(q["distractors"]) != 3:
                        raise ValueError(f"Question {i} must have exactly 3 distractors")
                    if not all(isinstance(d, str) for d in q["distractors"]):
                        raise ValueError(f"Question {i} distractors must be strings")

                    all_options = [q["correct_answer"]] + q["distractors"]
                    random.shuffle(all_options)
                    labels = ["A. ", "B. ", "C. ", "D. "]
                    labeled_options = [f"{labels[idx]}{opt}" for idx, opt in enumerate(all_options)]
                    correct_labeled = next(
                        opt for opt in labeled_options if opt.endswith(q["correct_answer"])
                    )

                    formatted_questions.append({
                        "question": q["question"],
                        "options": labeled_options,
                        "correct_answer": correct_labeled
                    })
                questions = formatted_questions
            return questions
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse Gemini response as JSON", error=str(e))
            raise GeminiUnavailableError(f"Gemini returned malformed response: {e}")

    @staticmethod
    def _is_multiple_choice(q: Union[str, dict, MultipleChoiceQuestion]) -> bool:
        """Return True if the question is a multiple‑choice item."""
        if isinstance(q, dict):
            return "options" in q and "correct_answer" in q
        if hasattr(q, "options") and hasattr(q, "correct_answer"):
            return True
        return False

    @staticmethod
    def _normalize_question(q: Union[str, dict, MultipleChoiceQuestion]) -> dict:
        """Convert any question representation into a uniform dict."""
        if isinstance(q, str):
            return {"question": q, "type": "single"}
        if isinstance(q, dict):
            result = dict(q)
            result["type"] = "multiple_choice" if "options" in q else "single"
            return result
        return {
            "question": q.question,
            "options": getattr(q, "options", None),
            "correct_answer": getattr(q, "correct_answer", None),
            "type": "multiple_choice" if hasattr(q, "options") else "single",
        }

    def _format_qa_pairs_for_prompt(
        self,
        questions: List[Union[str, dict, MultipleChoiceQuestion]],
        answers: List[str]
    ) -> str:
        """Build a detailed QA block that includes MC metadata."""
        formatted = []
        for i, (q_raw, ans) in enumerate(zip(questions, answers), start=1):
            q = self._normalize_question(q_raw)
            if q["type"] == "multiple_choice":
                options = q.get("options", [])
                correct = q.get("correct_answer", "")
                formatted.append(
                    f"--- Question {i} (Multiple‑Choice) ---\n"
                    f"Question: {q['question']}\n"
                    f"Options: {', '.join(options)}\n"
                    f"User Answer: {ans}\n"
                    f"Correct Answer: {correct}\n"
                )
            else:
                formatted.append(
                    f"--- Question {i} (Open‑Ended) ---\n"
                    f"Question: {q['question']}\n"
                    f"User Answer: {ans}\n"
                )
        return "\n".join(formatted)

    def _compute_mc_score(
        self,
        questions: List[Union[str, dict, MultipleChoiceQuestion]],
        answers: List[str]
    ) -> Optional[float]:
        """Compute multiple‑choice score as percentage. Returns None if no MC questions."""
        mc_correct = 0
        mc_total = 0
        for q_raw, ans in zip(questions, answers):
            q = self._normalize_question(q_raw)
            if q["type"] == "multiple_choice":
                mc_total += 1
                user_ans = str(ans).strip().lower()
                correct_ans = str(q.get("correct_answer", "")).strip().lower()

                if correct_ans.startswith(("a. ", "b. ", "c. ", "d. ")):
                    if user_ans and user_ans[0].lower() == correct_ans[0].lower():
                        mc_correct += 1
                elif user_ans == correct_ans:
                    mc_correct += 1
        if mc_total == 0:
            return None
        return (mc_correct / mc_total) * 100

    def _apply_authenticity_penalty(
        self,
        result: Dict[str, Any],
        hard_flag: bool,
        wps: float,
        time_taken: int,
        has_subjective: bool
    ) -> Dict[str, Any]:
        """
        Enforce authenticity rules programmatically.
        - If no subjective answers exist, ignore any suspicion (no typing involved).
        - If hard_flag is True, mark suspicious and apply penalty.
        - Otherwise, preserve LLM's suspicion flag but only apply penalty if it was set.
        """
        auth = result.get("authenticity_flag", {})
        is_suspicious = auth.get("is_suspicious", False)

        if not has_subjective:
            auth["is_suspicious"] = False
            auth["reason"] = "Pure multiple‑choice assessment (no typed answers)."
            result["authenticity_flag"] = auth
            return result

        if hard_flag:
            is_suspicious = True
            auth["is_suspicious"] = True
            auth["reason"] = (
                f"Completion time of {time_taken}s ({wps:.2f} words/sec) "
                f"exceeds human typing limit of {HUMAN_TYPING_SPEED_THRESHOLD_WPS} wps."
            )

        if is_suspicious:
            original_score = result.get("overall_score", 0)
            penalized = max(0, original_score - AUTHENTICITY_PENALTY)
            result["overall_score"] = penalized
            logger.info(
                "Authenticity penalty applied",
                original_score=original_score,
                penalized_score=penalized,
                reason=auth.get("reason")
            )

        result["authenticity_flag"] = auth
        return result

    @staticmethod
    def _extract_question_text(question: Union[str, dict]) -> str:
        """Extract the question text from a question object."""
        if isinstance(question, str):
            return question
        if isinstance(question, dict):
            return question.get("question", "")
        try:
            return question.question
        except AttributeError:
            pass
        raise TypeError(f"Unsupported question type: {type(question)}")

    def grade_answers(
        self,
        questions: List[Union[str, dict, MultipleChoiceQuestion]],
        answers: List[str],
        time_taken_seconds: int
    ) -> Dict[str, Any]:
        """
        Grade a candidate's answers and produce a score, skill breakdown, and authenticity flag.
        """
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
        if time_taken_seconds <= 0:
            raise ValueError("time_taken_seconds must be positive")

        logger.info(
            "Grading assessment answers",
            num_questions=len(questions),
            time_taken_seconds=time_taken_seconds,
        )

        mc_score = self._compute_mc_score(questions, answers)
        has_mc = mc_score is not None

        qa_pairs_block = self._format_qa_pairs_for_prompt(questions, answers)

        subjective_answers = []
        for q_raw, ans in zip(questions, answers):
            q = self._normalize_question(q_raw)
            if q["type"] == "single":
                subjective_answers.append(str(ans))

        has_subjective = len(subjective_answers) > 0
        if has_subjective:
            total_subjective_words = sum(len(ans.split()) for ans in subjective_answers)
            wps = total_subjective_words / time_taken_seconds if time_taken_seconds > 0 else 999.0
            hard_flag = wps > HUMAN_TYPING_SPEED_THRESHOLD_WPS
        else:
            wps = 0.0
            hard_flag = False

        if has_mc:
            grading_rules_block = f"""
The system has pre‑calculated the multiple‑choice score as **{mc_score:.1f}%** ({int(mc_score)}/100).
Use this as the foundation for the `overall_score`. If there are also subjective questions,
adjust the score up or down based on the quality of those answers.
Do NOT award 100% unless both MC and subjective answers are perfect.
"""
        else:
            grading_rules_block = """
All questions are open‑ended. Evaluate based on technical depth, clarity, and relevance.
"""

        prompt = GRADE_ANSWERS_PROMPT_TEMPLATE.format(
            time_taken_seconds=time_taken_seconds,
            wps=round(wps, 2),
            hard_flag="TRUE" if hard_flag else "FALSE",
            grading_rules_block=grading_rules_block,
            qa_pairs=qa_pairs_block
        )
        generated = self.gemini.generate(prompt)

        try:
            result = parse_gemini_json(generated)
        except json.JSONDecodeError as e:
            logger.error("Gemini returned non‑JSON response", error=str(e))
            raise GeminiUnavailableError(f"Malformed JSON from Gemini: {e}")

        required_keys = {"overall_score", "skill_breakdown", "authenticity_flag"}
        if not all(k in result for k in required_keys):
            raise ValueError("Gemini response missing required keys")

        overall = int(result["overall_score"])
        if not (0 <= overall <= 100):
            raise ValueError(f"overall_score {overall} out of range 0‑100")

        skill_breakdown = result["skill_breakdown"]
        if not isinstance(skill_breakdown, list) or not (3 <= len(skill_breakdown) <= 5):
            raise ValueError("skill_breakdown must be a list of 3‑5 items")
        for item in skill_breakdown:
            if not all(k in item for k in ("category", "score", "feedback")):
                raise ValueError("skill_breakdown item missing required keys")
            score = int(item["score"])
            if not (0 <= score <= 100):
                raise ValueError(f"Skill score {score} out of range 0‑100")

        if has_mc:
            llm_score = result["overall_score"]
            if not any(not self._is_multiple_choice(q) for q in questions):
                if abs(llm_score - mc_score) > 1:
                    logger.warning(
                        "LLM deviated from computed MC score, overriding",
                        llm_score=llm_score,
                        mc_score=mc_score
                    )
                    result["overall_score"] = int(round(mc_score))
            else:
                lower_bound = max(0, int(mc_score) - 30)
                upper_bound = min(100, int(mc_score) + 30)
                if not (lower_bound <= llm_score <= upper_bound):
                    logger.warning(
                        "LLM score outside reasonable range given MC score",
                        llm_score=llm_score,
                        mc_score=mc_score,
                        bounds=(lower_bound, upper_bound)
                    )
                    result["overall_score"] = max(lower_bound, min(llm_score, upper_bound))

        result = self._apply_authenticity_penalty(
            result, hard_flag, wps, time_taken_seconds, has_subjective
        )

        logger.info(
            "Grading completed",
            overall_score=result["overall_score"],
            is_suspicious=result["authenticity_flag"]["is_suspicious"]
        )
        return result