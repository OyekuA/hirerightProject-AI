

import json
import re
import unittest
from unittest.mock import MagicMock, patch
from app.services.assessment_service import AssessmentService
from app.clients.llm import LLMUnavailableError
from app.clients.dependencies import JOBS_COLLECTION, CANDIDATES_COLLECTION
from app.schemas.assessment import (
    CandidateContext,
    GenerateAssessmentRequest,
    GradeAssessmentResponse,
    JobContext,
)
from pydantic import ValidationError

class TestAssessmentService(unittest.TestCase):

    def setUp(self):

        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.service = AssessmentService(llm=self.gemini_mock, qdrant=self.qdrant_mock)

    def test_generate_questions_valid_response(self):

        candidate_payload = {"past_roles": ["Engineer at Acme"], "skills": ["Python"]}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1", "Q2", "Q3"]'
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=3,
        )
        self.assertEqual(result, ["Q1", "Q2", "Q3"])
        self.qdrant_mock.get.assert_called_once()
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("3", prompt)
        call_kwargs = self.gemini_mock.generate.call_args.kwargs
        self.assertIn("response_format", call_kwargs)
        self.assertIn("max_tokens", call_kwargs)

    def test_generate_questions_clamped_to_30(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = json.dumps([f"Q{i}" for i in range(1, 31)])
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=35,
        )
        self.assertEqual(len(result), 30)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("30", prompt)
        self.assertNotIn("35", prompt)

    def test_generate_questions_clamped_to_minimum_1(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1"]'
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Engineer",
            num_questions=0,
        )
        self.assertEqual(len(result), 1)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Generate exactly 1", prompt)
        self.assertNotIn("Generate exactly 0", prompt)

    def test_generate_questions_candidate_not_found(self):

        self.qdrant_mock.get.return_value = None
        with self.assertRaises(ValueError) as cm:
            self.service.generate_questions(
                candidate_id=999,
                target_role="Software Engineer",
                num_questions=3,
            )
        self.assertIn("Candidate not found", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_generate_questions_fewer_than_requested(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1"]'
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.generate_questions(
                candidate_id=42,
                target_role="Software Engineer",
                num_questions=3,
            )
        self.assertIn("only", str(cm.exception).lower())
        self.assertEqual(self.gemini_mock.generate.call_count, 2)

    def test_generate_questions_job_id_only(self):

        job_payload = {
            "title": "Software Engineer",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "We need a skilled engineer.",
        }
        self.qdrant_mock.get.return_value = job_payload
        self.gemini_mock.generate.return_value = '["Q1", "Q2", "Q3"]'
        result = self.service.generate_questions(
            candidate_id=None,
            target_role="Software Engineer",
            num_questions=3,
            job_id=99,
        )
        self.assertEqual(result, ["Q1", "Q2", "Q3"])
        self.qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 99)
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Software Engineer", prompt)

    def test_generate_questions_job_id_only_derived_target_role(self):

        job_payload = {
            "title": "Senior DevOps Engineer",
            "required_skills": ["Kubernetes", "AWS"],
            "raw_jd_summary": "We need a senior DevOps engineer.",
        }
        self.qdrant_mock.get.return_value = job_payload
        self.gemini_mock.generate.return_value = '["Q1", "Q2"]'
        result = self.service.generate_questions(
            candidate_id=None,
            target_role=None,
            num_questions=2,
            job_id=99,
        )
        self.assertEqual(result, ["Q1", "Q2"])
        self.qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 99)
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Senior DevOps Engineer", prompt)

    def test_generate_questions_job_id_empty_title_raises(self):

        job_payload = {"title": "", "required_skills": [], "raw_jd_summary": ""}
        self.qdrant_mock.get.return_value = job_payload
        with self.assertRaises(ValueError) as cm:
            self.service.generate_questions(
                candidate_id=None,
                target_role=None,
                num_questions=3,
                job_id=99,
            )
        self.assertIn("Could not determine target role", str(cm.exception))
        self.qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 99)
        self.gemini_mock.generate.assert_not_called()

    def test_generate_questions_combined_ids(self):

        candidate_payload = {"past_roles": ["Engineer at Acme"], "skills": ["Python"]}
        job_payload = {
            "title": "Senior Software Engineer",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Senior role.",
        }
        self.qdrant_mock.get.side_effect = lambda collection, idx: (
            candidate_payload if collection == CANDIDATES_COLLECTION else job_payload
        )
        self.gemini_mock.generate.return_value = '["Q1", "Q2"]'
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=2,
            job_id=99,
        )
        self.assertEqual(result, ["Q1", "Q2"])
        self.assertEqual(self.qdrant_mock.get.call_count, 2)
        self.qdrant_mock.get.assert_any_call(CANDIDATES_COLLECTION, 42)
        self.qdrant_mock.get.assert_any_call(JOBS_COLLECTION, 99)
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Engineer at Acme", prompt)
        self.assertIn("Senior Software Engineer", prompt)

    def test_generate_questions_multiple_choice_happy_path(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = json.dumps([
            {
                "question": "What would you do?",
                "correct_answer": "Option 1",
                "distractors": ["Option 2", "Option 3", "Option 4"],
            }
        ])
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = self.service.generate_questions(
                candidate_id=42,
                target_role="Software Engineer",
                num_questions=1,
                question_type="multiple_choice",
            )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(result[0]["question"], "What would you do?")
        self.assertEqual(len(result[0]["options"]), 4)
        self.assertEqual(result[0]["options"], ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"])
        self.assertEqual(result[0]["correct_answer"], "A. Option 1")
        self.gemini_mock.generate.assert_called_once()

    def test_generate_questions_malformed_multiple_choice_rejection(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = json.dumps([
            {"question": "Q1", "distractors": ["Option 2", "Option 3", "Option 4"]}
        ])
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.generate_questions(
                candidate_id=42,
                target_role="Software Engineer",
                num_questions=1,
                question_type="multiple_choice",
            )
        self.assertIn("malformed", str(cm.exception).lower())

    def test_generate_questions_no_ids_raises(self):

        with self.assertRaises(ValueError) as cm:
            self.service.generate_questions(
                candidate_id=None,
                target_role="Software Engineer",
                num_questions=3,
                job_id=None,
            )
        self.assertIn("at least one", str(cm.exception).lower())
        self.qdrant_mock.get.assert_not_called()
        self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_valid_response_single(self):

        questions = ["What is Python?", "Explain OOP"]
        answers = ["Python is a language.", "OOP is about objects."]
        expected_result = {
            "overall_score": 85,
            "skill_breakdown": [
                {"category": "Fundamentals", "score": 80, "feedback": "You show basic understanding."},
                {"category": "OOP", "score": 75, "feedback": "You explain OOP but lack depth."},
                {"category": "Communication", "score": 70, "feedback": "Clear but concise."}
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No suspicious indicators."},
            "grading_reasoning": "Score based on answer quality and completeness.",
        }
        self.gemini_mock.generate.return_value = json.dumps(expected_result)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=120,
        )
        self.assertEqual(result["overall_score"], 85)
        self.assertIsInstance(result["skill_breakdown"], list)
        self.assertTrue(3 <= len(result["skill_breakdown"]) <= 5)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("120", prompt)

    def test_grade_answers_valid_response_mc(self):

        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
            {"question": "Q2", "options": ["A. True", "B. False"], "correct_answer": "B. False"},
        ]
        answers = ["A. Yes", "B. False"]
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=1,
        )
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["skill_breakdown"], [])
        self.assertFalse(result["authenticity_flag"]["is_suspicious"])
        self.assertIn("Pure multiple-choice", result["authenticity_flag"]["reason"])
        self.assertFalse(result["needs_review"])
        self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_mixed_mc_and_single(self):

        questions = [
            {"question": "MC Q1", "options": ["A. 1", "B. 2"], "correct_answer": "A. 1"},
            "Explain your answer."
        ]
        answers = ["B. 2", "Because I think so."]
        mc_score = 0.0
        llm_response = {
            "overall_score": 60,
            "skill_breakdown": [
                {"category": "MC", "score": 0, "feedback": "You missed the MC."},
                {"category": "Subjective", "score": 60, "feedback": "Decent explanation."},
                {"category": "Overall", "score": 30, "feedback": "Needs improvement."}
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
            "grading_reasoning": "Mixed assessment with MC and subjective answers.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=60,
        )
        self.assertEqual(result["overall_score"], 30)

    def test_grade_answers_mismatched_length(self):

        questions = ["Q1", "Q2"]
        answers = ["A1"]
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=60)
        self.assertIn("must match", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_zero_or_negative_time(self):

        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        for invalid_time in (0, -5):
            with self.subTest(invalid_time=invalid_time):
                with self.assertRaises(ValueError) as cm:
                    self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=invalid_time)
                self.assertIn("must be positive", str(cm.exception))
                self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_malformed_json(self):

        questions = ["Q1"]
        answers = ["A1"]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(LLMUnavailableError):
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertEqual(self.gemini_mock.generate.call_count, 2)

    def test_grade_answers_skill_breakdown_too_short(self):

        questions = ["Q1"]
        answers = ["A1"]
        invalid_result = {
            "overall_score": 80,
            "skill_breakdown": [],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
            "grading_reasoning": "Some reasoning.",
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("malformed", str(cm.exception).lower())

    def test_grade_answers_skill_breakdown_malformed_item(self):

        questions = ["Q1"]
        answers = ["A1"]
        invalid_result = {
            "overall_score": 80,
            "skill_breakdown": [
                {"category": "A", "score": 70, "feedback": "Good"},
                {"category": "B", "score": 80, "feedback": "Great"},
                {"score": 90, "feedback": "Missing category"},  # no 'category'
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
            "grading_reasoning": "Some reasoning.",
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("malformed", str(cm.exception).lower())

    def test_grade_answers_overall_score_out_of_bounds(self):

        questions = ["Q1"]
        answers = ["A1"]
        invalid_result = {
            "overall_score": 150,
            "skill_breakdown": [
                {"category": "A", "score": 70, "feedback": "Good"},
                {"category": "B", "score": 80, "feedback": "Great"},
                {"category": "C", "score": 90, "feedback": "Excellent"},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
            "grading_reasoning": "Some reasoning.",
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("malformed", str(cm.exception).lower())

    def test_grade_answers_authenticity_penalty_applied(self):

        questions = ["Explain concurrency in Python."]
        answers = ["Concurrency can be achieved using threading or asyncio. The GIL limits true parallelism..."]  # long answer
        llm_response = {
            "overall_score": 90,
            "skill_breakdown": [
                {"category": "Concurrency", "score": 90, "feedback": "Good."},
                {"category": "Knowledge", "score": 90, "feedback": "Solid."},
                {"category": "Clarity", "score": 90, "feedback": "Clear."}
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
            "grading_reasoning": "Score based on answer quality.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=1,
        )
        self.assertEqual(result["overall_score"], 65)
        self.assertTrue(result["authenticity_flag"]["is_suspicious"])
        self.assertIn("exceeds human typing limit", result["authenticity_flag"]["reason"])

    def test_compute_mc_score_exact_match(self):

        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
        ]
        answers = ["A. Yes"]
        score = self.service._compute_mc_score(questions, answers)
        self.assertEqual(score, 100.0)

    def test_compute_mc_score_label_only_match(self):

        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
        ]
        answers = ["A"]
        score = self.service._compute_mc_score(questions, answers)
        self.assertEqual(score, 100.0)

    def test_compute_mc_score_label_lowercase_match(self):

        questions = [
            {"question": "Q1", "options": ["A. Opt 1", "B. Option 2"], "correct_answer": "B. Option 2"},
        ]
        answers = ["b"]
        score = self.service._compute_mc_score(questions, answers)
        self.assertEqual(score, 100.0)

    def test_compute_mc_score_wrong_label_does_not_match(self):

        questions = [
            {"question": "Q1", "options": ["A. Opt 1", "B. Option 2"], "correct_answer": "B. Option 2"},
        ]
        answers = ["A"]
        score = self.service._compute_mc_score(questions, answers)
        self.assertEqual(score, 0.0)

    def test_compute_mc_score_no_mc_returns_none(self):

        questions = ["Open-ended Q1", "Open-ended Q2"]
        answers = ["A1", "A2"]
        score = self.service._compute_mc_score(questions, answers)
        self.assertIsNone(score)

    def test_grade_answers_slow_typing_not_flagged_suspicious(self):

        questions = ["Explain microservices."]
        answers = ["A"]
        llm_response = {
            "overall_score": 80,
            "skill_breakdown": [
                {"category": "Knowledge", "score": 80, "feedback": "Good."},
                {"category": "Clarity", "score": 80, "feedback": "Clear."},
                {"category": "Depth", "score": 80, "feedback": "Adequate."},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No issues."},
            "grading_reasoning": "Score based on answer quality.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=600,
        )
        self.assertFalse(result["authenticity_flag"]["is_suspicious"])
        self.assertEqual(result["overall_score"], 80)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_grade_answers_bad_coherent_answers_score_above_zero(self):

        questions = ["Explain microservices."]
        answers = ["I think it means small services."]
        llm_response = {
            "overall_score": 15,
            "skill_breakdown": [
                {"category": "Knowledge", "score": 15, "feedback": "Partial."},
                {"category": "Clarity", "score": 15, "feedback": "Weak."},
                {"category": "Depth", "score": 15, "feedback": "Shallow."},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No issues."},
            "grading_reasoning": "Score based on answer quality.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=120,
        )
        self.assertEqual(result["overall_score"], 15)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("11–30", prompt)

    def test_grade_answers_pure_mc_skips_llm(self):

        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
        ]
        answers = ["A. Yes"]
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=30,
        )
        self.gemini_mock.generate.assert_not_called()
        self.assertEqual(result["overall_score"], 100)
        self.assertEqual(result["skill_breakdown"], [])
        self.assertFalse(result["authenticity_flag"]["is_suspicious"])
        self.assertFalse(result["needs_review"])

    def test_generate_questions_mc_embedded_dot_space_in_answer(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = json.dumps([
            {
                "question": "What framework should you use?",
                "correct_answer": "Use FastAPI. It scales well.",
                "distractors": ["Use Flask.", "Use Django.", "Use Node.js."],
            }
        ])
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = self.service.generate_questions(
                candidate_id=42,
                target_role="Software Engineer",
                num_questions=1,
                question_type="multiple_choice",
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["correct_answer"], "A. Use FastAPI. It scales well.")
        self.assertEqual(len(result[0]["options"]), 4)

    def test_grade_answers_pure_mc_includes_grading_reasoning(self):

        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
        ]
        answers = ["A. Yes"]
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=30,
        )
        self.assertIn("grading_reasoning", result)
        self.assertEqual(
            result["grading_reasoning"],
            "Pure MC assessment — score computed deterministically.",
        )

    def test_generate_questions_repair_retry_on_malformed(self):

        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        # First call returns malformed, second returns valid
        self.gemini_mock.generate.side_effect = [
            "not json",
            '["Q1", "Q2"]'
        ]
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=2,
        )
        self.assertEqual(result, ["Q1", "Q2"])
        self.assertEqual(self.gemini_mock.generate.call_count, 2)

    def test_grade_answers_llm_path_includes_grading_reasoning(self):

        questions = ["Explain OOP."]
        answers = ["OOP stands for Object-Oriented Programming."]
        llm_response = {
            "overall_score": 85,
            "skill_breakdown": [
                {"category": "Fundamentals", "score": 80, "feedback": "Good."},
                {"category": "Clarity", "score": 85, "feedback": "Clear."},
                {"category": "Depth", "score": 70, "feedback": "Needs more."},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No issues."},
            "grading_reasoning": "Solid understanding but lacks depth in some areas.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=120,
        )
        self.assertIn("grading_reasoning", result)
        self.assertEqual(
            result["grading_reasoning"],
            "Solid understanding but lacks depth in some areas.",
        )

    def test_grade_answers_llm_false_positive_suspicion_cleared_for_slow_typing(self):

        questions = ["Explain microservices."]
        answers = ["Microservices are small independent services that work together."]
        llm_response = {
            "overall_score": 80,
            "skill_breakdown": [
                {"category": "Knowledge", "score": 80, "feedback": "Good."},
                {"category": "Clarity", "score": 80, "feedback": "Clear."},
                {"category": "Depth", "score": 80, "feedback": "Adequate."},
            ],
            "authenticity_flag": {"is_suspicious": True, "reason": "LLM hallucinated suspicion."},
            "grading_reasoning": "Score based on answer quality.",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=600,  # ~0.02 wps — well below 2.5 threshold
        )
        self.assertFalse(result["authenticity_flag"]["is_suspicious"])
        self.assertEqual(result["overall_score"], 80)

    def test_generate_questions_produces_unique_prompts_on_each_call(self):

        candidate_payload = {"past_roles": ["Engineer at Acme"], "skills": ["Python", "FastAPI", "AWS", "Docker"]}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1", "Q2", "Q3"]'

        self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=3,
        )
        self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=3,
        )

        prompt_1 = self.gemini_mock.generate.call_args_list[0][0][0]
        prompt_2 = self.gemini_mock.generate.call_args_list[1][0][0]

        self.assertIn("candidate self-assessing", prompt_1)
        self.assertIn("candidate self-assessing", prompt_2)

        self.assertNotEqual(
            prompt_1, prompt_2,
            "Prompts must differ between calls to prevent repeated questions",
        )

class TestGenerateAssessmentRequestModel(unittest.TestCase):

    def test_both_contexts_accepted(self):
        candidate = CandidateContext(candidate_id=123, target_role="Engineer")
        job = JobContext(job_id=456)
        req = GenerateAssessmentRequest(
            candidate_context=candidate,
            job_context=job,
            num_questions=5,
            question_type="single"
        )
        self.assertEqual(req.candidate_context, candidate)
        self.assertEqual(req.job_context, job)

    def test_candidate_context_only_accepted(self):
        candidate = CandidateContext(candidate_id=123, target_role="Engineer")
        req = GenerateAssessmentRequest(
            candidate_context=candidate,
            job_context=None,
            num_questions=3,
            question_type="multiple_choice"
        )
        self.assertIsNone(req.job_context)

    def test_job_context_only_accepted(self):
        job = JobContext(job_id=456)
        req = GenerateAssessmentRequest(
            candidate_context=None,
            job_context=job,
            num_questions=2,
            question_type="single"
        )
        self.assertIsNone(req.candidate_context)

    def test_neither_context_raises(self):
        with self.assertRaises(ValidationError) as cm:
            GenerateAssessmentRequest(
                candidate_context=None,
                job_context=None,
                num_questions=1,
                question_type="single"
            )
        self.assertIn("At least one", str(cm.exception))

class TestGradeAssessmentResponseModel(unittest.TestCase):

    def test_model_validate_preserves_grading_reasoning(self):

        payload = {
            "overall_score": 85,
            "skill_breakdown": [
                {"category": "A", "score": 80, "feedback": "Good."},
                {"category": "B", "score": 85, "feedback": "Clear."},
                {"category": "C", "score": 70, "feedback": "Needs work."},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No issues."},
            "needs_review": False,
            "grading_reasoning": "Candidate shows solid fundamentals.",
        }
        response = GradeAssessmentResponse.model_validate(payload)
        self.assertEqual(response.grading_reasoning, "Candidate shows solid fundamentals.")
        serialized = response.model_dump()
        self.assertIn("grading_reasoning", serialized)
        self.assertEqual(serialized["grading_reasoning"], "Candidate shows solid fundamentals.")

    def test_model_validate_defaults_grading_reasoning_to_empty_string(self):

        payload = {
            "overall_score": 90,
            "skill_breakdown": [
                {"category": "A", "score": 90, "feedback": "Excellent."},
                {"category": "B", "score": 90, "feedback": "Excellent."},
                {"category": "C", "score": 90, "feedback": "Excellent."},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "Clean."},
            "needs_review": False,
        }
        response = GradeAssessmentResponse.model_validate(payload)
        self.assertEqual(response.grading_reasoning, "")
        serialized = response.model_dump()
        self.assertIn("grading_reasoning", serialized)
        self.assertEqual(serialized["grading_reasoning"], "")