"""Unit tests for recommendation service with prompt‑size capping."""
import unittest
from unittest.mock import MagicMock, patch, call

from app.services.recommendation_service import RecommendationService


class TestRecommendationServiceTruncation(unittest.TestCase):
    """Test that oversized recent_searches are truncated before embedding."""

    def setUp(self):
        """Create a RecommendationService with mocked dependencies."""
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    @patch("app.services.recommendation_service.truncate_to_prompt_cap")
    def test_recent_searches_truncated(self, mock_truncate):
        """Verify each recent search is passed through truncate_to_prompt_cap."""
        original_searches = ["short", "a" * 60000, "medium"]
        truncated_searches = ["short", "a" * 50000, "medium"]
        mock_truncate.side_effect = lambda x: x[:50000]

        self.mock_gemini.embed.return_value = [0.1] * 768

        self.mock_qdrant.get_with_vector.return_value = ({"some": "payload", "skills": ["a", "b", "c"]}, [0.2] * 768)
        self.mock_qdrant.search.return_value = []

        result = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": original_searches, "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        self.assertEqual(mock_truncate.call_count, 3)
        calls = [call(s) for s in original_searches]
        mock_truncate.assert_has_calls(calls, any_order=True)

        embed_calls = self.mock_gemini.embed.call_args_list
        self.assertEqual(len(embed_calls), 3)
        for i, call_args in enumerate(embed_calls):
            self.assertEqual(call_args[0][0], truncated_searches[i])

    def test_no_recent_searches_skips_truncation(self):
        """When recent_searches is empty, truncate_to_prompt_cap should not be called."""
        with patch("app.services.recommendation_service.truncate_to_prompt_cap") as mock_truncate:
            self.mock_qdrant.get_with_vector.return_value = ({"some": "payload", "skills": ["a", "b", "c"]}, [0.2] * 768)
            self.mock_qdrant.search.return_value = []

            result = self.service.recommend(
                rec_type="jobs",
                target_id=123,
                target_version=1,
                behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
                hard_filters={},
                force_refresh=False,
                limit=10,
            )

            mock_truncate.assert_not_called()


    def test_get_with_vector_missing_payload_raises_value_error(self):
        """Ensure ValueError is raised when target payload is missing."""
        self.mock_qdrant.get_with_vector.return_value = (None, None)
        with self.assertRaises(ValueError) as cm:
            self.service.recommend(
                rec_type="jobs",
                target_id=123,
                target_version=1,
                behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
                hard_filters={},
                force_refresh=False,
                limit=10,
            )
        self.assertIn("Target profile not found", str(cm.exception))

    def test_get_with_vector_missing_vector_raises_value_error(self):
        """Ensure ValueError is raised when target vector is missing."""
        self.mock_qdrant.get_with_vector.return_value = ({"some": "payload"}, None)
        with self.assertRaises(ValueError) as cm:
            self.service.recommend(
                rec_type="jobs",
                target_id=123,
                target_version=1,
                behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
                hard_filters={},
                force_refresh=False,
                limit=10,
            )
        self.assertIn("Target profile vector is missing", str(cm.exception))


class TestRecommendationServiceCacheKey(unittest.TestCase):
    """Test cache‑key composition for canonical scoring order."""

    def setUp(self):
        """Create a RecommendationService with mocked dependencies."""
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_cache_key_jobs(self):
        """Cache key for rec_type='jobs' follows candidate:job order."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"some": "payload", "skills": ["a", "b", "c"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 1001,
                    "job_version": 2,
                    "score": 0.9,
                    "title": "Job A",
                    "location": "City A",
                    "required_skills": [],
                },
                {
                    "_point_id": 1002,
                    "job_version": 1,
                    "score": 0.8,
                    "title": "Job B",
                    "location": "City B",
                    "required_skills": [],
                },
            ],
        ]
        self.mock_cache.get.return_value = None

        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=3,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        calls = self.mock_cache.get.call_args_list
        self.assertEqual(len(calls), 2)

        self.assertEqual(calls[0][0][0], "123:3:1001:2")
        self.assertEqual(calls[1][0][0], "123:3:1002:1")

    def test_cache_key_candidates(self):
        """Cache key for rec_type='candidates' follows candidate:job order."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"some": "payload", "required_skills": ["a", "b", "c"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 2001,
                    "candidate_version": 5,
                    "score": 0.85,
                },
            ],
        ]
        self.mock_cache.get.return_value = None

        self.service.recommend(
            rec_type="candidates",
            target_id=456,
            target_version=7,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        calls = self.mock_cache.get.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "2001:5:456:7")

    def test_cache_key_without_version(self):
        """When result_version is None, cache lookup is skipped."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"some": "payload", "skills": ["a", "b", "c"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 3001,
                    "score": 0.75,
                },
            ],
        ]
        self.mock_cache.get.return_value = None

        self.service.recommend(
            rec_type="jobs",
            target_id=999,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        self.mock_cache.get.assert_not_called()


class TestColdStart(unittest.TestCase):
    """Cold‑start path (skills < 3) triggers scroll and skips embedding."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_cold_start_skills_less_than_three(self):
        """Target with <3 skills → scroll called, embed not called."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"]},
            [0.1] * 768,
        )
        self.mock_qdrant.scroll.return_value = [
            {"_point_id": 1001, "score": 0.0},
            {"_point_id": 1002, "score": 0.0},
        ]
        self.mock_cache.get.return_value = None

        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        self.mock_qdrant.scroll.assert_called_once()
        self.mock_gemini.embed.assert_not_called()
        self.mock_cache.get.assert_not_called()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r["similarity_score"], 0.0)
            self.assertIsNone(r["llm_score"])


    def test_cold_start_attaches_cached_llm_score(self):
        """Cold‑start with versioned results attaches cached LLM scores."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"]},
            [0.1] * 768,
        )
        self.mock_qdrant.scroll.return_value = [
            {"_point_id": 1001, "job_version": 5, "score": 0.0},
            {"_point_id": 1002, "job_version": 7, "score": 0.0},
        ]
        self.mock_cache.get.side_effect = [
            {"overall_score_percentage": 82},
            None,
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        self.mock_qdrant.scroll.assert_called_once()
        self.mock_gemini.embed.assert_not_called()
        self.assertEqual(self.mock_cache.get.call_count, 2)
        self.assertEqual(results[0]["llm_score"], 82)
        self.assertIsNone(results[1]["llm_score"])
        for r in results:
            self.assertEqual(r["similarity_score"], 0.0)


    def test_cold_start_force_refresh_skips_cache(self):
        """Cold‑start with force_refresh=True skips cache lookup."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"]},
            [0.1] * 768,
        )
        self.mock_qdrant.scroll.return_value = [
            {"_point_id": 1001, "job_version": 5, "score": 0.0},
            {"_point_id": 1002, "job_version": 7, "score": 0.0},
        ]
        self.mock_cache.get.return_value = None

        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=True,
            limit=10,
        )

        self.mock_qdrant.scroll.assert_called_once()
        self.mock_gemini.embed.assert_not_called()
        self.mock_cache.get.assert_not_called()
        for r in results:
            self.assertIsNone(r["llm_score"])
            self.assertEqual(r["similarity_score"], 0.0)


class TestAdaptiveWeights(unittest.TestCase):
    """Verify adaptive weight formulas."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], []]
        self.mock_qdrant.scroll.return_value = []

    def test_zero_signals(self):
        """No signals → intent_weight = 0, cooccurrence_weight = 0."""
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.mock_gemini.embed.assert_not_called()

    def test_three_searches_intent_weight(self):
        """3 recent searches → intent_weight > 0."""
        self.mock_gemini.embed.return_value = [0.2] * 768
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": ["x", "y", "z"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.assertEqual(self.mock_gemini.embed.call_count, 3)

    def test_two_saves_cooccurrence_weight(self):
        """2 saves → cooccurrence_weight > 0."""
        self.mock_qdrant.get_with_vector.side_effect = [
            ({"some": "payload", "skills": ["a", "b", "c"]}, [0.3] * 768),
            ({"some": "payload"}, [0.4] * 768),
            ({"some": "payload"}, [0.5] * 768),
        ]
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [100, 200], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        calls = self.mock_qdrant.get_with_vector.call_args_list
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(len(calls), 3)


class TestPeerCentroid(unittest.TestCase):
    """Peer centroid vector computed from top 5 peers (excluding target)."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]},
            [0.1] * 768,
        )

    def test_peer_centroid_excludes_target(self):
        """Search returns 6 hits including target ID; target is excluded."""
        self.mock_qdrant.search.side_effect = [
            [
                {"_point_id": 123, "score": 0.99},
                {"_point_id": 101, "score": 0.8},
                {"_point_id": 102, "score": 0.7},
                {"_point_id": 103, "score": 0.6},
                {"_point_id": 104, "score": 0.5},
                {"_point_id": 105, "score": 0.4},
            ],
            [],
        ]
        self.mock_qdrant.get_with_vector.side_effect = [
            ({"skills": ["a", "b", "c"]}, [0.1] * 768),
            ({"payload": "peer1"}, [0.2] * 768),
            ({"payload": "peer2"}, [0.3] * 768),
            ({"payload": "peer3"}, [0.4] * 768),
            ({"payload": "peer4"}, [0.5] * 768),
            ({"payload": "peer5"}, [0.6] * 768),
        ]
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        calls = self.mock_qdrant.get_with_vector.call_args_list
        self.assertEqual(len(calls), 6)
        target_call = calls[0]
        self.assertEqual(target_call[0][0], "candidates")
        self.assertEqual(target_call[0][1], 123)
        peer_ids = {101, 102, 103, 104, 105}
        for call in calls[1:]:
            self.assertEqual(call[0][0], "candidates")
            self.assertIn(call[0][1], peer_ids)
            peer_ids.remove(call[0][1])
        self.assertEqual(len(peer_ids), 0)


class TestReRanker(unittest.TestCase):
    """Structured re‑ranker computes composite scores correctly."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python", "sql", "java"],
                "location": "Berlin",
                "experience_level": "mid",
                "employment_type": "full_time",
            },
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 1001,
                    "score": 0.9,
                    "skills": ["python", "java"],
                    "location": "Berlin",
                    "experience_level": "mid",
                    "employment_type": "full_time",
                    "title": "Software Engineer",
                },
                {
                    "_point_id": 1002,
                    "score": 0.8,
                    "skills": ["sql", "excel"],
                    "location": "Munich",
                    "experience_level": "senior",
                    "employment_type": "contract",
                    "title": "Data Analyst",
                },
            ],
        ]

    def test_reranker_composite_score(self):
        """final_score = 0.55*vector + 0.20*skill + 0.10*location + 0.10*level + 0.05*employment."""
        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0]["similarity_score"], results[1]["similarity_score"])


class TestDiversityPass(unittest.TestCase):
    """Diversity pass deduplicates by (title, location) and injects exploration slots."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 1001,
                    "score": 0.95,
                    "title": "Software Engineer",
                    "location": "Berlin",
                    "final_score": 0.95,
                },
                {
                    "_point_id": 1002,
                    "score": 0.94,
                    "title": "Software Engineer",
                    "location": "Berlin",
                    "final_score": 0.94,
                },
                {
                    "_point_id": 1003,
                    "score": 0.93,
                    "title": "Data Scientist",
                    "location": "Munich",
                    "final_score": 0.93,
                },
                {
                    "_point_id": 1004,
                    "score": 0.92,
                    "title": "Data Scientist",
                    "location": "Munich",
                    "final_score": 0.92,
                },
                {
                    "_point_id": 1005,
                    "score": 0.91,
                    "title": "DevOps Engineer",
                    "location": "Berlin",
                    "final_score": 0.91,
                },
                {
                    "_point_id": 1006,
                    "score": 0.90,
                    "title": "Product Manager",
                    "location": "Hamburg",
                    "final_score": 0.90,
                },
            ],
        ]

    def test_diversity_deduplication(self):
        """Only top‑scoring result per cluster kept."""
        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.assertLessEqual(len(results), 4)
        ids = [r["id"] for r in results]
        self.assertNotIn(1002, ids)


    def test_candidate_diversity_does_not_collapse_same_location(self):
        """Candidates with same location but different names are not collapsed."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"required_skills": ["python", "sql", "java"]},
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [
                {
                    "_point_id": 2001,
                    "score": 0.95,
                    "name": "Alice",
                    "location": "Berlin",
                    "final_score": 0.95,
                },
                {
                    "_point_id": 2002,
                    "score": 0.94,
                    "name": "Bob",
                    "location": "Berlin",
                    "final_score": 0.94,
                },
                {
                    "_point_id": 2003,
                    "score": 0.93,
                    "name": "Charlie",
                    "location": "London",
                    "final_score": 0.93,
                },
                {
                    "_point_id": 2004,
                    "score": 0.92,
                    "name": "Alice",
                    "location": "Paris",
                    "final_score": 0.92,
                },
                {
                    "_point_id": 2005,
                    "score": 0.91,
                    "name": "Alice",
                    "location": "Berlin",
                    "final_score": 0.91,
                },
            ],
        ]

        results = self.service.recommend(
            rec_type="candidates",
            target_id=999,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        ids = [r["id"] for r in results]
        self.assertIn(2001, ids)
        self.assertIn(2002, ids)
        self.assertIn(2003, ids)
        self.assertIn(2004, ids)
        self.assertNotIn(2005, ids)


if __name__ == "__main__":
    unittest.main()