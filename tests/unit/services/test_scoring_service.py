"""Unit tests for ScoringService."""

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.scoring_service import ScoringService
from app.clients.llm import LLMUnavailableError

VALID_RESULT = {
    "overall_score_percentage": 75,
    "category_breakdown": {
        "role_match": {"status": "pass", "short_reason": "Good match"},
        "experience": {"status": "warning", "short_reason": "Some gaps"},
        "location": {"status": "pass", "short_reason": "Same city"},
        "employment_type": {"status": "pass", "short_reason": "Full-time match"},
    },
    "skill_gap_analysis": "Candidate lacks Kubernetes experience.",
}


class TestScoringService(unittest.TestCase):
    """Test the ScoringService.calculate_fit method."""

    def setUp(self):
        """Create mocked dependencies for each test."""
        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.cache_mock = MagicMock()
        self.service = ScoringService(
            llm=self.gemini_mock,
            qdrant=self.qdrant_mock,
            cache=self.cache_mock,
        )
        self.settings_patcher = patch(
            "app.services.scoring_service.get_settings",
            return_value=MagicMock(CACHE_TTL_SECONDS=86400, MAX_PROMPT_CHARS=50000),
        )
        self.settings_patcher.start()
        self.truncate_patcher = patch(
            "app.services.scoring_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.settings_patcher.stop()
        self.truncate_patcher.stop()

    def test_cache_hit_skips_llm(self):
        """When cache contains a valid result, LLM and Qdrant should not be called."""
        self.cache_mock.get.return_value = VALID_RESULT
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result, VALID_RESULT)
        self.gemini_mock.generate.assert_not_called()
        self.cache_mock.get.assert_called_once_with("1:2:3:4")
        self.qdrant_mock.get.assert_not_called()

    def test_cache_miss_calls_llm(self):
        """Cache miss should trigger an LLM call and store the result."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result, VALID_RESULT)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once_with("1:2:3:4", VALID_RESULT, ttl=86400)

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True should ignore cache and call LLM."""
        self.cache_mock.get.return_value = VALID_RESULT
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=True,
        )
        self.assertEqual(result, VALID_RESULT)
        self.cache_mock.get.assert_not_called()
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once_with("1:2:3:4", VALID_RESULT, ttl=86400)

    def test_candidate_not_found_raises_value_error(self):
        """If candidate does not exist, raise ValueError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.return_value = None
        with self.assertRaises(ValueError) as cm:
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertIn("Candidate not found", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_job_not_found_raises_value_error(self):
        """If job does not exist, raise ValueError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            None,
        ]
        with self.assertRaises(ValueError) as cm:
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertIn("Job not found", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_malformed_llm_json_raises_error(self):
        """If LLM returns non‑JSON, raise json.JSONDecodeError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(json.JSONDecodeError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_llm_response_missing_required_keys_raises_error(self):
        """Valid JSON missing required keys raises LLMUnavailableError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps({"skill_gap_analysis": "some gap"})
        with self.assertRaises(LLMUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_overall_score_percentage_out_of_range_raises_error(self):
        """overall_score_percentage out of range raises LLMUnavailableError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        response = VALID_RESULT.copy()
        response["overall_score_percentage"] = 150
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(LLMUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_overall_score_percentage_non_integer_raises_error(self):
        """Non-integer overall_score_percentage raises LLMUnavailableError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        response = VALID_RESULT.copy()
        response["overall_score_percentage"] = "seventy-five"
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(LLMUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_cache_hit_returns_immediately_without_qdrant(self):
        """Cache hit should return the cached value immediately without Qdrant calls."""
        self.cache_mock.get.return_value = VALID_RESULT
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result, VALID_RESULT)
        self.qdrant_mock.get.assert_not_called()
        self.gemini_mock.generate.assert_not_called()
        self.cache_mock.get.assert_called_once_with("1:2:3:4")

    def test_different_version_causes_cache_miss(self):
        """A version change should produce a different cache key, resulting in a miss."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result, VALID_RESULT)
        self.cache_mock.get.assert_called_once_with("1:2:3:4")
        self.assertEqual(self.qdrant_mock.get.call_count, 2)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once_with("1:2:3:4", VALID_RESULT, ttl=86400)

    def test_cache_key_format(self):
        """Verify the cache key is built from the four IDs."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.cache_mock.get.assert_called_once_with("1:2:3:4")
        self.cache_mock.set.assert_called_once_with("1:2:3:4", VALID_RESULT, ttl=86400)

    def test_prompt_contains_no_pronoun_rule(self):
        """Verify the prompt includes the no‑pronoun rule."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("\"The candidate\"", prompt)
        self.assertIn("\"They\"", prompt)
        self.assertIn("\"Your\"", prompt)

    def test_prompt_short_reason_instruction(self):
        """Verify the prompt includes short‑reason and neutral match statement instructions."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("short_reason", prompt)
        self.assertIn("Telegraphic style", prompt)

    def test_prompt_skill_gap_analysis_instruction(self):
        """Verify the prompt includes skill_gap_analysis instruction."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("skill_gap_analysis", prompt)
        self.assertIn("most significant gaps", prompt)

    def test_candidate_version_mismatch_raises_value_error(self):
        """Candidate version mismatch should raise ValueError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2},
            {"job_version": 4},
        ]
        with self.assertRaises(ValueError) as cm:
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=3,  # mismatch
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertIn("Candidate version mismatch", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_job_version_mismatch_raises_value_error(self):
        """Job version mismatch should raise ValueError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2},
            {"job_version": 4},
        ]
        with self.assertRaises(ValueError) as cm:
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=5,  # mismatch
                force_refresh=False,
            )
        self.assertIn("Job version mismatch", str(cm.exception))
        self.gemini_mock.generate.assert_not_called()

    def test_calculate_fit_single_call_returns_score(self):
        """A single LLM call returns the overall_score_percentage directly."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Engineer"},
        ]
        score_data = dict(VALID_RESULT, overall_score_percentage=75)
        self.gemini_mock.generate.return_value = json.dumps(score_data)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result["overall_score_percentage"], 75)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_calculate_fit_malformed_json_raises_error(self):
        """Malformed JSON response raises json.JSONDecodeError from parse_llm_json."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Engineer"},
        ]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(json.JSONDecodeError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_calculate_fit_score_anchor_transferable_skills(self):
        """Prompt contains SCORING ANCHORS and the Do NOT score below 70 rule."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"candidate_version": 2, "name": "Alice"},
            {"job_version": 4, "title": "Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps(VALID_RESULT)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("SCORING RUBRIC", prompt)
        self.assertIn("Tier 2", prompt)