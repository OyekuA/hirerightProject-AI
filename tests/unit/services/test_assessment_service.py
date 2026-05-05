"""Unit tests for AssessmentService."""

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.assessment_service import AssessmentService
from app.clients.llm import LLMUnavailableError
from app.clients.dependencies import JOBS_COLLECTION, CANDIDATES_COLLECTION
from app.schemas.assessment import GenerateAssessmentRequest, CandidateContext, JobContext
from pydantic import ValidationError


class TestAssessmentService(unittest.TestCase):
    """Test the AssessmentService methods."""

    def setUp(self):
        """Create mocked dependencies for each test."""
        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.service = AssessmentService(llm=self.gemini_mock, qdrant=self.qdrant_mock)

    def test_generate_questions_valid_response(self):
        """generate_questions should return the list of questions from LLM."""
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

    def test_generate_questions_clamped_to_30(self):
        """Requested number of questions should be clamped to a maximum of 30."""
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
        """num_questions below 1 should be clamped to 1."""
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
        self.assertIn("1", prompt)
        self.assertNotIn("0", prompt)

    def test_generate_questions_candidate_not_found(self):
        """If candidate does not exist, raise ValueError."""
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
        """If LLM returns fewer questions than requested, raise LLMUnavailableError."""
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
        self.gemini_mock.generate.assert_called_once()

    def test_generate_questions_job_id_only(self):
        """generate_questions should work with only job_id provided."""
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
        """generate_questions should derive target_role from job title when target_role is None."""
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
        """generate_questions should raise ValueError when job title is empty and target_role not provided."""
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
        """generate_questions should fetch both candidate and job contexts."""
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
        """generate_questions should return multiple‑choice objects."""
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
        """generate_questions should reject malformed multiple‑choice payloads."""
        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        # Missing correct_answer key
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
        """generate_questions must have at least one ID."""
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
        """grade_answers should return the parsed JSON dict from LLM for single response."""
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
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("120", prompt)

    def test_grade_answers_valid_response_mc(self):
        """grade_answers should handle pure multiple-choice correctly."""
        questions = [
            {"question": "Q1", "options": ["A. Yes", "B. No"], "correct_answer": "A. Yes"},
            {"question": "Q2", "options": ["A. True", "B. False"], "correct_answer": "B. False"},
        ]
        answers = ["A. Yes", "B. False"]
        llm_response = {
            "overall_score": 100,
            "skill_breakdown": [
                {"category": "Logic", "score": 100, "feedback": "Perfect."},
                {"category": "Knowledge", "score": 100, "feedback": "Excellent."},
                {"category": "Speed", "score": 100, "feedback": "Fast."}
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=1,
        )
        self.assertEqual(result["overall_score"], 100)
        self.assertFalse(result["authenticity_flag"]["is_suspicious"])
        self.assertIn("Pure multiple‑choice", result["authenticity_flag"]["reason"])

    def test_grade_answers_mixed_mc_and_single(self):
        """grade_answers should handle a mix of MC and subjective."""
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
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_response)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=60,
        )
        self.assertEqual(result["overall_score"], 30)

    def test_grade_answers_mismatched_length(self):
        """If questions and answers length mismatch, raise ValueError."""
        questions = ["Q1", "Q2"]
        answers = ["A1"]
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=60)
        self.assertIn("must match", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_zero_or_negative_time(self):
        """If time_taken_seconds is zero or negative, raise ValueError."""
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        for invalid_time in (0, -5):
            with self.subTest(invalid_time=invalid_time):
                with self.assertRaises(ValueError) as cm:
                    self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=invalid_time)
                self.assertIn("must be positive", str(cm.exception))
                self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_malformed_json(self):
        """If LLM returns non‑JSON, raise LLMUnavailableError."""
        questions = ["Q1"]
        answers = ["A1"]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("malformed", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_grade_answers_skill_breakdown_too_short(self):
        """LLM returns a skill_breakdown with fewer than three items."""
        questions = ["Q1"]
        answers = ["A1"]
        invalid_result = {
            "overall_score": 80,
            "skill_breakdown": [
                {"category": "A", "score": 70, "feedback": "Good"},
                {"category": "B", "score": 80, "feedback": "Great"},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "OK"},
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("must be a list of 3‑5 items", str(cm.exception))

    def test_grade_answers_skill_breakdown_malformed_item(self):
        """LLM returns a skill_breakdown with a missing key."""
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
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("missing required keys", str(cm.exception))

    def test_grade_answers_overall_score_out_of_bounds(self):
        """LLM returns an overall_score outside 0‑100."""
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
        }
        self.gemini_mock.generate.return_value = json.dumps(invalid_result)
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(questions=questions, answers=answers, time_taken_seconds=30)
        self.assertIn("out of range 0‑100", str(cm.exception))

    def test_grade_answers_authenticity_penalty_applied(self):
        """When typing speed exceeds limit, penalty is applied."""
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


class TestGenerateAssessmentRequestModel(unittest.TestCase):
    """Test the GenerateAssessmentRequest Pydantic model validation."""

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