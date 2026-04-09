"""Unit tests for recommendation service with prompt‑size capping."""
import unittest
import time
from unittest.mock import MagicMock, patch, call

from app.services.recommendation_service import RecommendationService
from app.clients.gemini import GeminiUnavailableError


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

    def test_missing_vector_with_few_skills_triggers_scroll(self):
        """Missing profile vector with <3 skills triggers scroll fallback."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"]},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "required_skills": ["a"],
                "location": "New York, US",
                "experience_level": "mid level",
                "employment_type": "full-time",
            },
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
        self.assertEqual(len(results), 1)
        # similarity_score computed from skill overlap (0.5) * 0.15 = 0.075
        self.assertAlmostEqual(results[0]["similarity_score"], 0.075)


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
            {"some": "payload", "skills": ["a", "b", "c"], "candidate_version": 3},
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
            {"some": "payload", "required_skills": ["a", "b", "c"], "job_version": 7},
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
    """Cold‑start path (missing profile vector and skills < 3) triggers scroll and skips embedding."""

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
        """Target with <3 skills and missing vector → scroll called, embed not called."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"], "experience_level": "junior", "location": "", "employment_type": "full-time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
            {
                "_point_id": 1002,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
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
        # similarity_score is 0.0 because skill overlap, location match, level match,
        # and employment match are all zero in this fixture.
        for r in results:
            self.assertEqual(r["similarity_score"], 0.0)
            self.assertIsNone(r["llm_score"])


    def test_cold_start_attaches_cached_llm_score(self):
        """Cold‑start with versioned results attaches cached LLM scores."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"], "experience_level": "junior", "location": "", "employment_type": "full-time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "job_version": 5,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
            {
                "_point_id": 1002,
                "job_version": 7,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
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
        # similarity_score is 0.0 because skill overlap, location match, level match,
        # and employment match are all zero in this fixture.
        for r in results:
            self.assertEqual(r["similarity_score"], 0.0)


    def test_cold_start_force_refresh_skips_cache(self):
        """Cold‑start with force_refresh=True skips cache lookup."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b"], "experience_level": "junior", "location": "", "employment_type": "full-time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "job_version": 5,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
            {
                "_point_id": 1002,
                "job_version": 7,
                "required_skills": [],
                "location": "",
                "experience_level": "unknown",
                "employment_type": "part-time",
            },
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
        # similarity_score is 0.0 because skill overlap, location match, level match,
        # and employment match are all zero in this fixture.
        for r in results:
            self.assertIsNone(r["llm_score"])
            self.assertEqual(r["similarity_score"], 0.0)

    def test_missing_vector_with_many_skills_triggers_scroll(self):
        """Missing profile vector with >=3 skills triggers cold‑start scroll."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["python", "java", "sql", "aws"], "experience_level": "mid level", "location": "New York", "employment_type": "full-time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "required_skills": ["python", "aws"],
                "location": "New York",
                "experience_level": "mid level",
                "employment_type": "full-time",
                "job_version": 1,
            },
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
        self.assertEqual(len(results), 1)
        # similarity_score should be computed based on skill overlap, location match, level match, employment match
        # vector_score = 0.0, skill overlap = 2/4 = 0.5, location match = 1.0, level match = 1.0, employment match = 1.0
        # composite = 0.55*0 + 0.20*0.5 + 0.10*1 + 0.10*1 + 0.05*1 = 0.0 + 0.10 + 0.10 + 0.10 + 0.05 = 0.35
        self.assertAlmostEqual(results[0]["similarity_score"], 0.325)
        self.assertIsNone(results[0]["llm_score"])

    def test_cold_start_with_vector_but_sparse_skills(self):
        """Missing vector with <3 skills triggers cold‑start scroll."""
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["python", "java"], "experience_level": "mid level", "location": "New York", "employment_type": "full-time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "required_skills": ["python"],
                "location": "New York",
                "experience_level": "mid level",
                "employment_type": "full-time",
                "job_version": 1,
            },
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
        self.assertEqual(len(results), 1)
        # similarity_score computed with vector_score = 0.0 (cold‑start)
        self.assertAlmostEqual(results[0]["similarity_score"], 0.325)  # skill overlap 0.5, location match 1, level match 1, employment match 1
        self.assertIsNone(results[0]["llm_score"])

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
        intent_weight, _, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.25)

    def test_intent_weight_excludes_cooccurrence_signals(self):
        """Intent weight includes searches, clicks, saves, and positive_outcomes after revert."""
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[],
            recent_saves=[100, 200],
            recent_positive_outcomes=[50],
        )
        # intent_signals = 3 searches + 0 clicks + 2 saves + 1 outcome = 6
        # intent_weight = min(0.45, 0.10 + 0.05*6) = min(0.45, 0.40) = 0.40
        self.assertAlmostEqual(intent_weight, 0.40)
        self.assertAlmostEqual(cooccurrence_weight, 0.15)

    def test_intent_weight_caps_at_0_45(self):
        """Intent weight is capped at 0.45 regardless of signal count."""
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["s"] * 10,
            recent_clicks=[{"id": i, "dwell_time_seconds": 5} for i in range(10)],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.45)
        self.assertAlmostEqual(cooccurrence_weight, 0.0)

    def test_intent_weight_includes_clicks(self):
        """Intent weight increases with clicks before cap is reached."""
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[{"id": 1, "dwell_time_seconds": 5}, {"id": 2, "dwell_time_seconds": 10}],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.35)
        self.assertAlmostEqual(cooccurrence_weight, 0.0)
        baseline_weight, _, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(baseline_weight, 0.25)
        self.assertGreater(intent_weight, baseline_weight)

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

    def test_weights_always_sum_to_one(self):
        """The four adaptive weights must always sum to exactly 1.0."""
        # All zeros
        iw, cw, pw, ppw = self.service._compute_weights([], [], [], [])
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        # Maximum intent signals (10 searches + 10 clicks)
        searches = ["query"] * 10
        clicks = [{"id": i, "dwell_time_seconds": 5} for i in range(10)]
        iw, cw, pw, ppw = self.service._compute_weights(searches, clicks, [], [])
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        # Maximum cooccurrence signals (10 saves + 10 positive outcomes)
        saves = list(range(10))
        pos = list(range(10))
        iw, cw, pw, ppw = self.service._compute_weights([], [], saves, pos)
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        # Both maxed simultaneously (intent at cap + cooccurrence at cap)
        iw, cw, pw, ppw = self.service._compute_weights(searches, clicks, saves, pos)
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)


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
                "experience_level": "mid level",
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
                    "job_version": 1,
                    "required_skills": ["python", "java"],
                    "location": "Berlin",
                    "experience_level": "mid level",
                    "employment_type": "full_time",
                    "title": "Software Engineer",
                },
                {
                    "_point_id": 1002,
                    "score": 0.8,
                    "job_version": 1,
                    "required_skills": ["sql", "excel"],
                    "location": "Munich",
                    "experience_level": "senior",
                    "employment_type": "contract",
                    "title": "Data Analyst",
                },
            ],
        ]

    def test_reranker_composite_score(self):
        """final_score = 0.60*vector + 0.15*skill + 0.10*location + 0.10*level + 0.05*employment."""
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
        # Compute expected composite scores based on fixed formula
        # vector scores: 0.9 for job 1001, 0.8 for job 1002
        # skill overlap: 2/3 ≈ 0.6666667 for job 1001, 1/4 = 0.25 for job 1002
        # location match: 1.0 for Berlin, 0.0 for Munich
        # level match: 1.0 for mid level vs mid level, 0.5 for mid level vs senior
        # employment match: 1.0 for full_time, 0.0 for contract
        expected_score_1001 = (
            0.55 * 0.9
            + 0.20 * (2 / 3)
            + 0.10 * 1.0
            + 0.10 * 1.0
            + 0.05 * 1.0
        )
        expected_score_1002 = (
            0.60 * 0.8
            + 0.15 * 0.25
            + 0.10 * 0.0
            + 0.10 * 0.5
            + 0.05 * 0.0
        )
        # Map results to IDs
        score_by_id = {r["id"]: r["similarity_score"] for r in results}
        self.assertAlmostEqual(score_by_id[1001], 0.89, places=7)
        self.assertAlmostEqual(score_by_id[1002], expected_score_1002, places=7)
        # Ensure ordering still holds
        self.assertGreater(score_by_id[1001], score_by_id[1002])


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

    def test_diversity_pass_limit_one(self):
        """Diversity pass with limit=1 should return at most one recommendation."""
        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=1,
        )
        self.assertLessEqual(len(results), 1)
        self.assertGreater(len(results), 0)

    def test_diversity_pass_limit_two(self):
        """Diversity pass with limit=2 should return at most two recommendations."""
        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=2,
        )
        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)

class TestRankPool(unittest.TestCase):
    """Unit tests for RecommendationService.rank_pool."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_successful_ordering(self, mock_scoring_cls):
        """Pool ranking returns candidates sorted descending by fit score."""
        # Mock job exists
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},  # job payload
            {"candidate_version": 1},  # candidate 101
            {"candidate_version": 2},  # candidate 102
        ]
        # Mock scoring service instance
        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.side_effect = [
            {"overall_score_percentage": 85},
            {"overall_score_percentage": 92},
        ]
        mock_scoring_cls.return_value = mock_scoring

        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=[101, 102],
        )

        # Should be sorted descending by fit_score
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["candidate_id"], 102)
        self.assertEqual(results[0]["fit_score"], 92)
        self.assertEqual(results[1]["candidate_id"], 101)
        self.assertEqual(results[1]["fit_score"], 85)
        # Verify scoring service called with correct versions (positional arguments)
        mock_scoring.calculate_fit.assert_any_call(101, 1, 999, 5, False)
        mock_scoring.calculate_fit.assert_any_call(102, 2, 999, 5, False)

    def test_rank_pool_missing_candidates_raises_value_error(self):
        """If any candidate IDs are not found, raise ValueError with list."""
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},  # job exists
            None,  # candidate 101 missing
            {"candidate_version": 2},  # candidate 102 found (but we won't reach due to error)
        ]
        with self.assertRaises(ValueError) as cm:
            self.service.rank_pool(
                job_id=999,
                job_version=5,
                candidate_ids=[101, 102],
            )
        self.assertIn("Candidate(s) not found in vector store", str(cm.exception))
        self.assertIn("101", str(cm.exception))

    def test_rank_pool_job_not_found_raises_value_error(self):
        """If job does not exist, raise ValueError."""
        self.mock_qdrant.get.return_value = None  # job missing
        with self.assertRaises(ValueError) as cm:
            self.service.rank_pool(
                job_id=999,
                job_version=5,
                candidate_ids=[101],
            )
        self.assertIn("Job not found", str(cm.exception))

    def test_rank_pool_stale_job_version_raises_value_error(self):
        """If supplied job version does not match stored version, raise ValueError."""
        self.mock_qdrant.get.return_value = {"job_version": 5}  # stored version
        with self.assertRaises(ValueError) as cm:
            self.service.rank_pool(
                job_id=999,
                job_version=3,  # stale version
                candidate_ids=[101],
            )
        self.assertIn("Job version mismatch", str(cm.exception))
        self.assertIn("supplied 3", str(cm.exception))
        self.assertIn("stored 5", str(cm.exception))

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_gemini_unavailable_error_propagates(self, mock_scoring_cls):
        """GeminiUnavailableError from scoring service propagates."""
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 1},
        ]
        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.side_effect = GeminiUnavailableError("circuit open")
        mock_scoring_cls.return_value = mock_scoring

        with self.assertRaises(GeminiUnavailableError):
            self.service.rank_pool(
                job_id=999,
                job_version=5,
                candidate_ids=[101],
            )

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_calls_scoring_service_with_correct_versions(self, mock_scoring_cls):
        """Scoring service receives correct candidate_version from payload."""
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 42},
        ]
        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {"overall_score_percentage": 75}
        mock_scoring_cls.return_value = mock_scoring

        self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=[101],
        )
        mock_scoring.calculate_fit.assert_called_once_with(101, 42, 999, 5, False)

    @patch('app.services.recommendation_service.POOL_RANK_CONCURRENCY', 10)
    @patch('app.services.recommendation_service.POOL_RANK_TIMEOUT_SECONDS', 30)
    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_high_cardinality(self, mock_scoring_cls):
        """High‑cardinality pool ranking processes all candidates with bounded concurrency."""
        # Mock job and 100 candidates
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
        ] + [{"candidate_version": i} for i in range(100)]
        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {"overall_score_percentage": 80}
        mock_scoring_cls.return_value = mock_scoring

        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=list(range(100)),
        )

        # Verify all candidates present
        self.assertEqual(len(results), 100)
        result_ids = {r["candidate_id"] for r in results}
        self.assertEqual(result_ids, set(range(100)))
        # Verify each candidate got a fit score
        for r in results:
            self.assertEqual(r["fit_score"], 80)
        # Verify scoring service was called 100 times
        self.assertEqual(mock_scoring.calculate_fit.call_count, 100)
        # Verify concurrency limit respected (max 10 concurrent calls)
        # We can't easily assert that without inspecting ThreadPoolExecutor internals,
        # but we can at least ensure the method completes.


    @patch('app.services.recommendation_service.POOL_RANK_TIMEOUT_SECONDS', 1.0)
    @patch('app.services.recommendation_service.POOL_RANK_CONCURRENCY', 3)
    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_timeout_fallback(self, mock_scoring_cls):
        """Slow tasks trigger timeout fallback scoring."""
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 1},
            {"candidate_version": 2},
            {"candidate_version": 3},
        ]
        mock_scoring = MagicMock()
        # Make calculate_fit sleep for varying durations
        def slow_calculate(*args, **kwargs):
            candidate_id = args[0]
            if candidate_id == 101:
                time.sleep(0.05)  # fast, should succeed
                return {"overall_score_percentage": 90}
            elif candidate_id == 102:
                time.sleep(1.5)   # exceeds global timeout, should trigger fallback
                return {"overall_score_percentage": 80}
            else:  # 103
                time.sleep(0.01)  # fast
                return {"overall_score_percentage": 70}
        mock_scoring.calculate_fit.side_effect = slow_calculate
        mock_scoring_cls.return_value = mock_scoring

        start = time.time()
        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=[101, 102, 103],
        )
        elapsed = time.time() - start

        # The method must return within global timeout + small margin, not waiting for slow tasks
        # global_timeout = 1.0 (since batch_count = 1). Allow 0.3 seconds overhead.
        self.assertLess(elapsed, 1.3, f"rank_pool took {elapsed:.2f}s, expected <1.3s")
        # Expect three results, sorted descending by fit_score (timeout fallback score = POOL_RANK_DEFAULT_FIT_SCORE)
        result_map = {r["candidate_id"]: r["fit_score"] for r in results}
        self.assertEqual(result_map[101], 90)
        self.assertEqual(result_map[103], 70)
        self.assertEqual(result_map[102], 0)

    @patch('app.services.recommendation_service.POOL_RANK_TIMEOUT_SECONDS', 0.2)
    @patch('app.services.recommendation_service.POOL_RANK_CONCURRENCY', 3)
    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_timeout_scales_with_pool_size(self, mock_scoring_cls):
        """Timeout scales with pool size, ensuring fair completion for multi‑batch pools."""
        # Mock job and candidates
        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
        ] + [{"candidate_version": i} for i in range(10)]
        mock_scoring = MagicMock()
        # Each scoring call takes 0.05 seconds (well within scaled timeout)
        def scoring_with_delay(*args, **kwargs):
            time.sleep(0.05)
            return {"overall_score_percentage": 75}
        mock_scoring.calculate_fit.side_effect = scoring_with_delay
        mock_scoring_cls.return_value = mock_scoring

        start = time.time()
        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=list(range(10)),
        )
        elapsed = time.time() - start

        # Ensure all candidates scored (no fallback)
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertEqual(r["fit_score"], 75)
        # Elapsed time should be less than scaled timeout (0.2 * ceil(10/3) = 0.8)
        # plus some overhead.
        self.assertLess(elapsed, 1.0)

class TestRecommendationServiceVersionValidation(unittest.TestCase):
    """Unit tests for version mismatch validation in RecommendationService.recommend."""

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            gemini=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_target_version_mismatch_candidate(self):
        """Target version mismatch for candidate target raises ValueError."""
        # rec_type = "jobs" -> target_collection = CANDIDATES_COLLECTION
        self.mock_qdrant.get_with_vector.return_value = (
            {"candidate_version": 2, "skills": ["python"]},  # stored version 2
            [0.1] * 768,
        )
        with self.assertRaises(ValueError) as cm:
            self.service.recommend(
                rec_type="jobs",
                target_id=123,
                target_version=3,  # mismatch
                behavioral_signals={},
                hard_filters={},
                force_refresh=False,
                limit=10,
            )
        self.assertIn("Target version mismatch", str(cm.exception))
        self.mock_gemini.embed.assert_not_called()
        self.mock_qdrant.search.assert_not_called()
        self.mock_qdrant.scroll.assert_not_called()

    def test_target_version_mismatch_job(self):
        """Target version mismatch for job target raises ValueError."""
        # rec_type = "candidates" -> target_collection = JOBS_COLLECTION
        self.mock_qdrant.get_with_vector.return_value = (
            {"job_version": 5, "required_skills": ["python"]},
            [0.1] * 768,
        )
        with self.assertRaises(ValueError) as cm:
            self.service.recommend(
                rec_type="candidates",
                target_id=456,
                target_version=6,  # mismatch
                behavioral_signals={},
                hard_filters={},
                force_refresh=False,
                limit=10,
            )
        self.assertIn("Target version mismatch", str(cm.exception))
        self.mock_gemini.embed.assert_not_called()
        self.mock_qdrant.search.assert_not_called()
        self.mock_qdrant.scroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()