"""Assessment service for generating interview questions and grading candidate answers.

This module provides the AssessmentService class that orchestrates Gemini calls
and Qdrant lookups to produce scenario‑based questions and evaluate answers.
"""

import json
import random
import structlog
from typing import Optional, Union, Literal

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils.ingestion import truncate_to_prompt_cap
from app.utils import parse_gemini_json
from app.prompts import GENERATE_QUESTIONS_PROMPT_TEMPLATE, GRADE_ANSWERS_PROMPT_TEMPLATE

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
        self,
        candidate_id: Optional[int],
        target_role: Optional[str],
        num_questions: int,
        job_id: Optional[int] = None,
        question_type: Literal["single", "multiple_choice"] = "single",
    ) -> list[Union[str, dict]]:
        """Generate scenario‑based interview questions tailored to a candidate or a job.

        Args:
            candidate_id: Unique identifier of the candidate in the vector store (optional).
            target_role: The role the candidate is being assessed for. If not provided, will be derived from the job title (requires job_id).
            num_questions: Desired number of questions (clamped to [1, 30]).
            job_id: Unique identifier of the job in the vector store (optional).
            question_type: Whether to generate single open‑ended questions or multiple‑choice questions.

        Returns:
            A list of exactly N questions. For "single" type, each item is a string.
            For "multiple_choice" type, each item is a dict with keys "question", "options", "correct_answer".

        Raises:
            ValueError: If neither candidate_id nor job_id is provided, or if the requested entity is not found.
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
        """
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
            raise ValueError("Could not determine target role — provide it explicitly or ensure the job has a title")

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
4. CRITICAL FORMATTING: The options must ONLY contain the proposed action or solution. DO NOT include explanations, justifications, or critiques inside the option text itself. 
   - BAD OPTION: "Query the monolith database directly. This creates tight coupling and impacts performance."
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
                    
                    correct_labeled = next(opt for opt in labeled_options if opt.endswith(q["correct_answer"]))
                    
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
    def _extract_question_text(question: Union[str, dict]) -> str:
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
        self, questions: list[Union[str, dict, "MultipleChoiceQuestion"]], answers: list[str], time_taken_seconds: int
    ) -> dict:
        """Grade a candidate's answers and produce a score, skill breakdown, and authenticity flag.

        Args:
            questions: List of questions that were asked. Each item can be a plain string, a multiple‑choice dict (must contain a 'question' key), or a MultipleChoiceQuestion Pydantic model.
            answers: List of answers provided by the candidate (same order as questions).
            time_taken_seconds: Total time the candidate spent answering all questions.

        Returns:
            A dictionary with the following keys:
                - overall_score (int 0‑100)
                - skill_breakdown (list of dicts each with keys category, score, feedback)
                - authenticity_flag (dict with keys is_suspicious and reason)

        Raises:
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
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

        question_texts = [self._extract_question_text(q) for q in questions]

        truncated_answers = [truncate_to_prompt_cap(ans) for ans in answers]
        qa_pairs = "\n".join(
            f"Question {i+1}: {q}\nAnswer {i+1}: {a}"
            for i, (q, a) in enumerate(zip(question_texts, truncated_answers))
        )

        mcq_total = 0
        mcq_correct = 0
        for q, a in zip(questions, answers):
            is_mcq = False
            sys_ans = ""
            if isinstance(q, dict):
                is_mcq = True
                sys_ans = str(q.get("correct_answer", "")).strip().lower()
            elif hasattr(q, "correct_answer"):
                is_mcq = True
                sys_ans = str(q.correct_answer).strip().lower()
            if is_mcq:
                mcq_total += 1
                user_ans = str(a).strip().lower()
                if sys_ans and sys_ans == user_ans:
                    mcq_correct += 1

        if mcq_total > 0:
            system_calculated_score = int((mcq_correct / mcq_total) * 100)
            grading_rules_block = f"""
1. **Pre-Calculated Math:** The backend system has already mathematically calculated the Multiple Choice score. It is exactly {system_calculated_score}%.
2. **Forcing the Score:** If the System Calculated MCQ Score is 100%, the `overall_score` MUST be exactly 100, and EVERY category in the `skill_breakdown` MUST be 100. Do not alter this math.
3. If the score is less than 100%, distribute the score across the `skill_breakdown` categories based on the specific questions the candidate missed, ensuring the final `overall_score` closely matches {system_calculated_score}%.
            """
        else:
            system_calculated_score = None
            grading_rules_block = """
1. **Open-Ended Evaluation:** Evaluate the candidate's open-ended answers based on technical accuracy, depth of knowledge, and problem-solving ability.
2. **Scoring:** Assign an `overall_score` between 0 and 100 based on the aggregate quality of the answers.
3. **Skill Distribution:** Break down the evaluation into specific skills and assign appropriate scores (0-100) for each category based on the candidate's demonstrated proficiency.
            """

        total_words = sum(len(a.split()) for a in answers)
        wps = total_words / time_taken_seconds if time_taken_seconds > 0 else 999.0
        is_mathematically_impossible = wps > 2.5

        prompt = GRADE_ANSWERS_PROMPT_TEMPLATE.format(
            time_taken_seconds=time_taken_seconds,
            wps=round(wps, 2),
            hard_flag="TRUE" if is_mathematically_impossible else "FALSE",
            grading_rules_block=grading_rules_block.strip(),
            qa_pairs=qa_pairs
        )
        
        generated = self.gemini.generate(prompt)
        try:
            result = parse_gemini_json(generated)
            if system_calculated_score is not None:
                result["overall_score"] = system_calculated_score
                
                if system_calculated_score == 100:
                    for item in result.get("skill_breakdown", []):
                        item["score"] = 100
            required_keys = {"overall_score", "skill_breakdown", "authenticity_flag"}
            if not all(k in result for k in required_keys):
                raise ValueError("Missing required keys in Gemini response")
            auth_flag = result["authenticity_flag"]
            if not isinstance(auth_flag, dict) or "is_suspicious" not in auth_flag or "reason" not in auth_flag:
                raise ValueError("authenticity_flag is malformed")
            overall_score = int(result["overall_score"])
            if not (0 <= overall_score <= 100):
                raise ValueError(f"overall_score {overall_score} is out of bounds (0‑100)")
            result["overall_score"] = overall_score
            skill_breakdown = result.get("skill_breakdown")
            if not isinstance(skill_breakdown, list) or not (3 <= len(skill_breakdown) <= 5):
                raise ValueError("skill_breakdown must be a list of 3‑5 items")
            for idx, item in enumerate(skill_breakdown):
                if not isinstance(item, dict):
                    raise ValueError(f"skill_breakdown[{idx}] must be a dict")
                if not all(k in item for k in ("category", "score", "feedback")):
                    raise ValueError(f"skill_breakdown[{idx}] missing required keys")
                item_score = item.get("score")
                if not isinstance(item_score, int) or not (0 <= item_score <= 100):
                    raise ValueError(f"skill_breakdown[{idx}].score must be int 0‑100")
            return result
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error("Failed to parse Gemini grading response", error=str(e))
            raise GeminiUnavailableError(f"Gemini returned malformed grading response: {e}")