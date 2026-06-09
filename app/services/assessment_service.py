import json
import random
import structlog
from typing import Optional, Union, Literal, List, Dict, Any

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient
from app.utils import parse_llm_json
from app.prompts import GENERATE_QUESTIONS_PROMPT_TEMPLATE, GRADE_ANSWERS_PROMPT_TEMPLATE
from ..schemas.assessment import MultipleChoiceQuestion

logger = structlog.get_logger()

HUMAN_TYPING_SPEED_THRESHOLD_WPS = 2.5
AUTHENTICITY_PENALTY = 25

GENERIC_COMPETENCY_AXES = (
    "core role-specific expertise",
    "analytical problem solving",
    "communication and collaboration",
    "ownership, judgement, and execution",
    "adaptability and continuous learning",
)

ASSESSMENT_FOCUS_ANGLES = (
    "designing a solution or system and the trade-offs it requires",
    "diagnosing and resolving an unexpected failure or incident",
    "improving performance, scalability, or efficiency",
    "managing security, privacy, compliance, or risk",
    "working with data quality, modelling, metrics, or measurement",
    "aligning stakeholders with competing priorities",
    "mentoring, coaching, or upskilling a colleague",
    "operating under ambiguous or shifting requirements",
    "making trade-offs under tight time or resource constraints",
    "defining a quality, testing, or validation strategy",
    "improving a process or automating a workflow",
    "resolving conflict or disagreement on a team",
    "navigating an ethical dilemma or difficult judgement call",
    "planning and executing a migration, rollout, or major change",
    "collaborating across functions with a different discipline",
    "deciding under uncertainty or with incomplete information",
    "learning from a mistake, failure, or post-mortem",
    "evaluating and adopting a new tool, method, or technology",
    "balancing short-term delivery against long-term sustainability",
    "monitoring, observing, or measuring impact after delivery",
    "ramping up quickly in an unfamiliar domain or system",
    "driving alignment or leading without formal authority",
    "responding to direct customer or end-user impact",
    "doing more with constrained budget or resources",
)

ASSESSMENT_SITUATION_MODIFIERS = (
    "a tight deadline or high-pressure environment",
    "limited budget, headcount, or tooling",
    "a newly formed, distributed, or junior team",
    "significant legacy systems or accumulated technical debt",
    "a period of rapid growth or a sudden spike in demand",
    "conflicting expectations from multiple stakeholders",
    "a regulated or low-tolerance-for-error context",
)


class AssessmentService:

    def __init__(self, llm: LLMClient, qdrant: Optional[QdrantClient] = None):
        self.llm = llm
        self.qdrant = qdrant

    def generate_questions(
        self,
        candidate_id: Optional[int],
        target_role: Optional[str],
        num_questions: int,
        job_id: Optional[int] = None,
        question_type: Literal["single", "multiple_choice"] = "single",
    ) -> List[Union[str, dict]]:
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
                "company_name": job_payload.get("company_name", ""),
                "location": job_payload.get("location", ""),
                "experience_level": job_payload.get("experience_level", ""),
                "employment_type": job_payload.get("employment_type", ""),
                "industry": job_payload.get("industry", ""),
                "about": job_payload.get("about", ""),
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
            _title = job_context.get("title", "").strip()
            _company_name = job_context.get("company_name", "").strip()
            _location = job_context.get("location", "").strip()
            _employment_type = job_context.get("employment_type", "").strip()
            _experience_level = job_context.get("experience_level", "").strip()
            _industry = job_context.get("industry", "").strip()
            _required_skills = [s.strip() for s in job_context.get("required_skills", []) if s.strip()]
            _raw_jd_summary = job_context.get("raw_jd_summary", "").strip()
            _about = job_context.get("about", "").strip()

            lines = ["The assessment is for the following role:"]
            if _title:
                lines.append(f"- Job Title: {_title}")
            if _company_name:
                lines.append(f"- Company: {_company_name}")
            if _location:
                lines.append(f"- Location: {_location}")
            if _employment_type:
                lines.append(f"- Employment Type: {_employment_type}")
            if _experience_level:
                lines.append(f"- Experience Level: {_experience_level}")
            if _industry:
                lines.append(f"- Industry: {_industry}")
            if _required_skills:
                lines.append(f"- Required Skills: {', '.join(_required_skills)}")
            if _raw_jd_summary:
                lines.append(f"- Role Summary: {_raw_jd_summary}")
            if _about:
                lines.append(f"- About the Company: {_about}")
            job_block = "\n".join(lines)
        else:
            job_block = """
The job context is not specified (generic role)."""

        skill_pool = list(dict.fromkeys(
            s.strip()
            for s in (skills + job_context.get("required_skills", []))
            if isinstance(s, str) and s.strip()
        ))
        if not skill_pool:
            skill_pool = list(GENERIC_COMPETENCY_AXES)

        focus_combos = [
            (skill, angle, situation)
            for skill in skill_pool
            for angle in ASSESSMENT_FOCUS_ANGLES
            for situation in ASSESSMENT_SITUATION_MODIFIERS
        ]
        random.shuffle(focus_combos)

        selected_focus = []
        seen_angles = set()
        for combo in focus_combos:
            if len(selected_focus) >= clamped_n:
                break
            if combo[1] not in seen_angles:
                seen_angles.add(combo[1])
                selected_focus.append(combo)
        if len(selected_focus) < clamped_n:
            for combo in focus_combos:
                if len(selected_focus) >= clamped_n:
                    break
                if combo not in selected_focus:
                    selected_focus.append(combo)

        focus_block = "\n".join(
            f"- Question {idx}: anchor it to **{skill}**, probed through the lens of "
            f"**{angle}**, situated in the context of **{situation}**."
            for idx, (skill, angle, situation) in enumerate(selected_focus, start=1)
        )

        if candidate_id is not None and job_id is None:
            intent_block = (
                "These questions are for a candidate self-assessing their fit for the role. "
                "Frame the scenarios as realistic 'day-in-the-life' situations to help them "
                "reflect on whether they have the right skills and temperament for this job."
            )
        else:
            intent_block = (
                "These questions are for an employer rigorously testing a candidate. "
                "Frame the scenarios to uncover the depth of the candidate's actual competency, "
                "problem-solving abilities, and past experience."
            )

        if question_type == "single":
            instruction_base = f"""
Generate exactly {clamped_n} scenario‑based questions that:
1. Are tailored to the specific industry, role, and skills provided.
2. Pose concrete, realistic work situations they will encounter in the {target_role} role.
3. Require the respondent to explain their strategy, thought process, or application of past experience.
4. Forbid generic definition questions (e.g., "What is X?") or trivia.

Return a JSON array of strings, where each string is one question, and nothing else.
"""
        else:
            instruction_base = f"""
Generate exactly {clamped_n} multiple-choice questions that:
1. Are tailored to the specific industry, role, and skills provided.
2. Pose highly complex, realistic work situations for a {target_role}.
3. The distractors (wrong options) MUST be common industry misconceptions, plausible mistakes, or sub-optimal strategies.
4. CRITICAL FORMATTING: The options must ONLY contain the proposed action or solution. Do NOT include explanations inside the option text itself.

Return a JSON array of objects, where each object has the following keys:
- "question": a string containing the scenario-based question.
- "correct_answer": a string with the exact correct answer.
- "distractors": an array of exactly 3 strings containing the plausible wrong answers.
Do NOT prepend A, B, C, D to the answers."""

        instruction = instruction_base + f"""

MANDATORY FOCUS ASSIGNMENTS — NON-NEGOTIABLE:
Produce the {clamped_n} questions strictly in order. Each question MUST be built around its assigned focus below, so that every question targets a genuinely different competency and scenario:

{focus_block}

ADDITIONAL RULES:
1. Treat each assignment as the seed for a concrete, realistic scenario a {target_role} would plausibly face — never a definition or trivia question.
2. Do NOT default to the most obvious textbook example for an assigned focus; pick a specific, less clichéd but still realistic situation.
3. Never mention, quote, label, or hint at these focus assignments, lenses, or situations in the question text — each question must read naturally and stand alone.
4. No two questions may collapse into the same underlying theme, even loosely.
5. Output strictly the requested JSON format and absolutely nothing else.
"""

        prompt = GENERATE_QUESTIONS_PROMPT_TEMPLATE.format(
            target_role=target_role,
            job_block=job_block,
            candidate_block=candidate_block,
            intent_block=intent_block,
            instruction=instruction
        )
        generated = self.llm.generate(prompt, temperature=0.9)
        try:
            parsed = parse_llm_json(generated)
            if not isinstance(parsed, list):
                raise ValueError("LLM response is not a list")
            questions = parsed
            if len(questions) > clamped_n:
                logger.warning(
                    "LLM returned more questions than requested, truncating",
                    requested=clamped_n,
                    received=len(questions),
                )
                questions = questions[:clamped_n]
            elif len(questions) < clamped_n:
                logger.error(
                    "LLM returned fewer questions than requested",
                    requested=clamped_n,
                    received=len(questions),
                )
                raise LLMUnavailableError(
                    f"LLM returned only {len(questions)} questions, expected {clamped_n}"
                )

            if question_type == "single":
                for q in questions:
                    if not isinstance(q, str):
                        raise ValueError("LLM response contains non‑string items in single‑question mode")
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
                    correct_labeled = None
                    for opt in labeled_options:
                        if opt[3:].strip() == q["correct_answer"].strip():
                            correct_labeled = opt
                            break
                    if correct_labeled is None:
                        raise ValueError(
                            f"Could not match correct answer '{q['correct_answer']}' among labeled options"
                        )

                    formatted_questions.append({
                        "question": q["question"],
                        "options": labeled_options,
                        "correct_answer": correct_labeled
                    })
                questions = formatted_questions
            return questions
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e))
            raise LLMUnavailableError(f"LLM returned malformed response: {e}")

    @staticmethod
    def _is_multiple_choice(q: Union[str, dict, MultipleChoiceQuestion]) -> bool:
        if isinstance(q, dict):
            return "options" in q and "correct_answer" in q
        if hasattr(q, "options") and hasattr(q, "correct_answer"):
            return True
        return False

    @staticmethod
    def _normalize_question(q: Union[str, dict, MultipleChoiceQuestion]) -> dict:
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
        mc_correct = 0
        mc_total = 0
        for q_raw, ans in zip(questions, answers):
            q = self._normalize_question(q_raw)
            if q["type"] == "multiple_choice":
                mc_total += 1
                user_ans = str(ans).strip().lower()
                correct_ans = str(q.get("correct_answer", "")).strip().lower()

                if user_ans == correct_ans:
                    mc_correct += 1
                elif len(user_ans) == 1 and user_ans.isalpha() and correct_ans.startswith(user_ans + ". "):
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
        else:
            auth["is_suspicious"] = False
            is_suspicious = False

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
        if isinstance(question, str):
            return question
        if isinstance(question, dict):
            return question.get("question", "")
        try:
            return question.question
        except AttributeError:
            pass
        raise TypeError(f"Unsupported question type: {type(question)}")

    def _run_grading(self, prompt: str) -> dict:
        generated = self.llm.generate(prompt, temperature=0)
        result = parse_llm_json(generated)

        required_keys = {"overall_score", "skill_breakdown", "authenticity_flag", "grading_reasoning"}
        if not all(k in result for k in required_keys):
            raise ValueError("LLM response missing required keys")
        if not isinstance(result["authenticity_flag"], dict):
            raise ValueError("authenticity_flag must be a dict")

        overall = int(result["overall_score"])
        if not (0 <= overall <= 100):
            raise ValueError(f"overall_score {overall} out of range 0‑100")
        result["overall_score"] = overall

        skill_breakdown = result["skill_breakdown"]
        if not isinstance(skill_breakdown, list) or not (3 <= len(skill_breakdown) <= 5):
            raise ValueError("skill_breakdown must be a list of 3‑5 items")
        for item in skill_breakdown:
            if not all(k in item for k in ("category", "score", "feedback")):
                raise ValueError("skill_breakdown item missing required keys")
            score = int(item["score"])
            if not (0 <= score <= 100):
                raise ValueError(f"Skill score {score} out of range 0‑100")
            item["score"] = score

        return result

    def grade_answers(
        self,
        questions: List[Union[str, dict, MultipleChoiceQuestion]],
        answers: List[str],
        time_taken_seconds: int
    ) -> Dict[str, Any]:
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

        if has_mc and not has_subjective:
            logger.info(
                "Pure MC assessment — skipping LLM vote, using deterministic score",
                mc_score=mc_score,
            )
            return {
                "overall_score": int(round(mc_score)),
                "skill_breakdown": [],
                "authenticity_flag": {
                    "is_suspicious": False,
                    "reason": "Pure multiple-choice assessment (no typed answers).",
                },
                "needs_review": False,
                "grading_reasoning": "Pure MC assessment — score computed deterministically.",
            }

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
        result = self._run_grading(prompt)

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

        result["needs_review"] = result["authenticity_flag"]["is_suspicious"]

        logger.info(
            "Grading completed",
            overall_score=result["overall_score"],
            is_suspicious=result["authenticity_flag"]["is_suspicious"],
            needs_review=result["needs_review"],
        )
        return result
