"""Unit tests for CareerPathService."""

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.career_service import CareerPathService
from app.clients.gemini import GeminiUnavailableError
from app.clients.dependencies import CANDIDATES_COLLECTION

VALID_ITEM = {
    "role": "Senior Data Engineer",
    "match_percentage": 75,
    "core_skills": ["Python", "SQL", "ETL"],
    "reasoning": "Your experience with Python and ETL pipelines aligns well with this role.",
}

VALID_PROFILE_SUMMARY = "You bring a strong foundation in backend systems and have demonstrated ownership of end‑to‑end data pipelines."

VALID_RESPONSE = json.dumps({
    "profile_summary": VALID_PROFILE_SUMMARY,
    "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]
})


class TestCareerPathService(unittest.TestCase):
    """Test the CareerPathService.analyze_career_paths method."""

    def setUp(self):
        """Create mocked dependencies for each test."""
        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.service = CareerPathService(
            gemini=self.gemini_mock,
            qdrant=self.qdrant_mock,
        )
        self.truncate_patcher = patch(
            "app.services.career_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.truncate_patcher.stop()

    def test_happy_path_returns_three_items(self):
        """analyze_career_paths should return a dict with profile_summary and paths."""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertIn("profile_summary", result)
        self.assertIn("paths", result)
        self.assertEqual(result["profile_summary"], VALID_PROFILE_SUMMARY)
        self.assertEqual(len(result["paths"]), 3)
        for item in result["paths"]:
            self.assertEqual(set(item.keys()), {"role", "match_percentage", "core_skills", "reasoning"})
            self.assertEqual(item["role"], "Senior Data Engineer")
            self.assertEqual(item["match_percentage"], 75)
            self.assertEqual(item["core_skills"], ["Python", "SQL", "ETL"])
        self.qdrant_mock.get.assert_called_once_with(CANDIDATES_COLLECTION, 42)
        self.gemini_mock.generate.assert_called_once()

    def test_candidate_not_found_raises_value_error(self):
        """If candidate does not exist, raise ValueError."""
        self.qdrant_mock.get.return_value = None
        with self.assertRaises(ValueError) as cm:
            self.service.analyze_career_paths(candidate_id=999)
        self.assertIn("not found", str(cm.exception).lower())
        self.gemini_mock.generate.assert_not_called()

    def test_gemini_non_json_raises_error(self):
        """If Gemini returns non‑JSON, raise GeminiUnavailableError."""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.assertIn("malformed", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_not_a_list_raises_error(self):
        """Item with core_skills as a string raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = "Python"
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_empty_list_raises_error(self):
        """Item with empty core_skills list raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = []
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_non_string_item_raises_error(self):
        """Item with non‑string skill raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["Python", 123, "SQL"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_too_few_raises_error(self):
        """Item with fewer than three skills raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["Python", "SQL"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_too_many_raises_error(self):
        """Item with more than five skills raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["a", "b", "c", "d", "e", "f"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_profile_summary_missing_raises_error(self):
        """Missing top-level profile_summary key raises GeminiUnavailableError."""
        response = {
            "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]
        }
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_profile_summary_empty_raises_error(self):
        """Empty top-level profile_summary string raises GeminiUnavailableError."""
        response = {
            "profile_summary": "",
            "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]
        }
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_reasoning_missing_raises_error(self):
        """Item missing reasoning key raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        del mutated["reasoning"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_reasoning_empty_raises_error(self):
        """Item with empty reasoning string raises GeminiUnavailableError."""
        mutated = VALID_ITEM.copy()
        mutated["reasoning"] = ""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_prompt_contains_second_person_instruction(self):
        """The generated prompt must contain second‑person instructions."""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("second", prompt.lower())
        self.assertIn("you", prompt.lower())
        self.assertIn("your", prompt.lower())

    def test_prompt_contains_core_skills_description(self):
        """The generated prompt must describe core_skills."""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("core_skills", prompt)

    def test_prompt_contains_profile_summary_description(self):
        """The generated prompt must describe profile_summary."""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("profile_summary", prompt)