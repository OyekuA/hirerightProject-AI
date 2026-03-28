"""Unit tests for AssessmentService."""

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.assessment_service import AssessmentService
from app.clients.gemini import GeminiUnavailableError


class TestAssessmentService(unittest.TestCase):
    """Test the AssessmentService methods."""

    def setUp(self):
        """Create mocked dependencies for each test."""
        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.service = AssessmentService(gemini=self.gemini_mock, qdrant=self.qdrant_mock)
        self.truncate_patcher = patch(
            "app.services.assessment_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.truncate_patcher.stop()

    def test_generate_questions_valid_response(self):
        """generate_questions should return the list of questions from Gemini."""
        candidate_payload = {
            "past_roles": ["Engineer at Acme"],
            "skills": ["Python"],
        }
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

    def test_generate_questions_clamped_to_5(self):
        """Requested number of questions should be clamped to a maximum of 5."""
        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1", "Q2", "Q3", "Q4", "Q5"]'
        result = self.service.generate_questions(
            candidate_id=42,
            target_role="Software Engineer",
            num_questions=10,
        )
        self.assertEqual(len(result), 5)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("5", prompt)
        self.assertNotIn("10", prompt)

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
        """If Gemini returns fewer questions than requested, raise GeminiUnavailableError."""
        candidate_payload = {"past_roles": [], "skills": []}
        self.qdrant_mock.get.return_value = candidate_payload
        self.gemini_mock.generate.return_value = '["Q1"]'
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.generate_questions(
                candidate_id=42,
                target_role="Software Engineer",
                num_questions=3,
            )
        self.assertIn("only", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_grade_answers_valid_response(self):
        """grade_answers should return the parsed JSON dict from Gemini."""
        questions = ["What is Python?", "Explain OOP"]
        answers = ["Python is a language.", "OOP is about objects."]
        expected_result = {
            "overall_score": 85,
            "feedback": "Good answers.",
            "authenticity_flag": {"is_suspicious": False, "reason": "Answers seem genuine."},
        }
        self.gemini_mock.generate.return_value = json.dumps(expected_result)
        result = self.service.grade_answers(
            questions=questions,
            answers=answers,
            time_taken_seconds=120,
        )
        self.assertEqual(result, expected_result)
        self.gemini_mock.generate.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("120", prompt)

    def test_grade_answers_mismatched_length(self):
        """If questions and answers length mismatch, raise ValueError."""
        questions = ["Q1", "Q2"]
        answers = ["A1"]
        with self.assertRaises(ValueError) as cm:
            self.service.grade_answers(
                questions=questions,
                answers=answers,
                time_taken_seconds=60,
            )
        self.assertIn("must match", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_grade_answers_malformed_json(self):
        """If Gemini returns non‑JSON, raise GeminiUnavailableError."""
        questions = ["Q1"]
        answers = ["A1"]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.grade_answers(
                questions=questions,
                answers=answers,
                time_taken_seconds=30,
            )
        self.assertIn("malformed", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()