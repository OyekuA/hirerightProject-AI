"""Unit tests for ScoringService."""

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.scoring_service import ScoringService
from app.clients.gemini import GeminiUnavailableError

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
            gemini=self.gemini_mock,
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

    def test_cache_hit_skips_gemini(self):
        """When cache contains a valid result, Gemini should not be called."""
        self.cache_mock.get.return_value = VALID_RESULT
        self.qdrant_mock.get.side_effect = [{"name": "Alice", "candidate_version": 2}, {"name": "Software Engineer", "job_version": 4}]
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
        self.assertEqual(self.qdrant_mock.get.call_count, 2)

    def test_cache_miss_calls_gemini(self):
        """Cache miss should trigger a Gemini call and store the result."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"name": "Alice"},
            {"title": "Software Engineer"},
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
        self.gemini_mock.generate.assert_called_once()
        self.cache_mock.set.assert_called_once_with("1:2:3:4", VALID_RESULT, ttl=86400)

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True should ignore cache and call Gemini."""
        self.cache_mock.get.return_value = VALID_RESULT
        self.qdrant_mock.get.side_effect = [
            {"name": "Alice"},
            {"title": "Software Engineer"},
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
        self.gemini_mock.generate.assert_called_once()
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
            {"name": "Alice"},
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

    def test_malformed_gemini_json_raises_error(self):
        """If Gemini returns non‑JSON, raise GeminiUnavailableError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"name": "Alice"},
            {"title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(GeminiUnavailableError) as cm:
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertIn("malformed", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_gemini_response_missing_required_keys_raises_error(self):
        """Valid JSON missing required keys raises GeminiUnavailableError."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"name": "Alice"},
            {"title": "Software Engineer"},
        ]
        self.gemini_mock.generate.return_value = json.dumps({"skill_gap_analysis": "some gap"})
        with self.assertRaises(GeminiUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.gemini_mock.generate.assert_called_once()

    def test_stale_cache_deleted_candidate(self):
        """When cache hit but candidate no longer exists, delete stale entry."""
        self.cache_mock.get.return_value = VALID_RESULT
        self.qdrant_mock.get.side_effect = [
            None,
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
        self.assertIn("Candidate not found", str(cm.exception))
        self.cache_mock.delete.assert_called_once_with("1:2:3:4")
        self.gemini_mock.generate.assert_not_called()

    def test_cache_key_format(self):
        """Verify the cache key is built from the four IDs."""
        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            {"name": "Alice"},
            {"title": "Software Engineer"},
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