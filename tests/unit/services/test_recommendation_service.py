import unittest
import time
from unittest.mock import MagicMock, patch, call

from app.services.recommendation_service import RecommendationService
from app.clients.llm import LLMUnavailableError

class TestRecommendationServiceTruncation(unittest.TestCase):

    def setUp(self):

        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    @patch("app.services.recommendation_service.truncate_to_prompt_cap")
    def test_recent_searches_truncated(self, mock_truncate):

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
        self.assertEqual(len(results), 0)
class TestRecommendationServiceCacheKey(unittest.TestCase):

    def setUp(self):

        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_cache_key_jobs(self):

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
        self.mock_cache.get.assert_not_called()
    def test_cache_key_candidates(self):

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
        self.mock_cache.get.assert_not_called()
    def test_cache_key_without_version(self):

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

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_cold_start_skills_less_than_three(self):

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
        self.assertEqual(len(results), 0)  # hard floor 0.50 drops all below-floor scroll results
    def test_cold_start_skips_cache_lookup(self):
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
        self.mock_cache.get.assert_not_called()
        self.assertEqual(len(results), 0)  # cache removed; hard floor drops all below-floor scroll results
    def test_cold_start_force_refresh_skips_cache(self):

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
        self.assertEqual(len(results), 0)  # hard floor drops all below-floor scroll results

    def test_missing_vector_with_many_skills_triggers_scroll(self):

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
        self.assertEqual(len(results), 0)  # composite 0.35 < hard floor 0.50
    def test_cold_start_with_vector_but_sparse_skills(self):

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
        self.assertEqual(len(results), 0)  # composite 0.35 < hard floor 0.50
class TestAdaptiveWeights(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
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

        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[],
            recent_saves=[100, 200],
            recent_positive_outcomes=[50],
        )
        self.assertAlmostEqual(intent_weight, 0.40)
        self.assertAlmostEqual(cooccurrence_weight, 0.15)

    def test_intent_weight_caps_at_0_45(self):

        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["s"] * 10,
            recent_clicks=[{"id": i, "dwell_time_seconds": 5} for i in range(10)],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.45)
        self.assertAlmostEqual(cooccurrence_weight, 0.20)

    def test_intent_weight_includes_clicks(self):

        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[{"id": 1, "dwell_time_seconds": 5}, {"id": 2, "dwell_time_seconds": 10}],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.35)
        self.assertAlmostEqual(cooccurrence_weight, 0.10)
        baseline_weight, _, _, _ = self.service._compute_weights(
            recent_searches=["x", "y", "z"],
            recent_clicks=[],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(baseline_weight, 0.25)
        self.assertGreater(intent_weight, baseline_weight)

    def test_two_saves_cooccurrence_weight(self):

        self.mock_qdrant._client = MagicMock()
        self.mock_qdrant._client.retrieve.return_value = [
            MagicMock(vector=[0.4] * 768),
            MagicMock(vector=[0.5] * 768),
        ]
        self.mock_qdrant.get_with_vector.return_value = (
            {"some": "payload", "skills": ["a", "b", "c"]}, [0.3] * 768,
        )
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [100, 200], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.mock_qdrant.get_with_vector.assert_called_once()
        self.mock_qdrant._client.retrieve.assert_called_once()

    def test_weights_always_sum_to_one(self):

        iw, cw, pw, ppw = self.service._compute_weights([], [], [], [])
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        searches = ["query"] * 10
        clicks = [{"id": i, "dwell_time_seconds": 5} for i in range(10)]
        iw, cw, pw, ppw = self.service._compute_weights(searches, clicks, [], [])
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        saves = list(range(10))
        pos = list(range(10))
        iw, cw, pw, ppw = self.service._compute_weights([], [], saves, pos)
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

        iw, cw, pw, ppw = self.service._compute_weights(searches, clicks, saves, pos)
        self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

class TestPeerCentroid(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]},
            [0.1] * 768,
        )

    def test_peer_centroid_excludes_target(self):

        self.mock_qdrant._client = MagicMock()
        self.mock_qdrant._client.retrieve.return_value = [
            MagicMock(vector=[0.2] * 768),
            MagicMock(vector=[0.3] * 768),
            MagicMock(vector=[0.4] * 768),
            MagicMock(vector=[0.5] * 768),
            MagicMock(vector=[0.6] * 768),
        ]
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
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, [0.1] * 768,
        )
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.mock_qdrant.get_with_vector.assert_called_once()
        self.mock_qdrant._client.retrieve.assert_called_once()
        retrieve_call = self.mock_qdrant._client.retrieve.call_args
        self.assertIn("ids", retrieve_call.kwargs)
        self.assertEqual(set(retrieve_call.kwargs["ids"]), {101, 102, 103, 104, 105})

class TestReRanker(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
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
        scaled_09 = (0.9 - 0.50) / 0.40
        scaled_08 = (0.8 - 0.50) / 0.40
        expected_score_1001 = (
            0.55 * scaled_09
            + 0.35 * (2 / 3)
            + 0.04 * 1.0
            + 0.04 * 1.0
            + 0.02 * 1.0
        )
        expected_score_1002 = (
            0.55 * scaled_08
            + 0.35 * 0.25
            + 0.04 * 0.0
            + 0.04 * 0.5
            + 0.02 * 0.0
        )
        score_by_id = {r["id"]: r["similarity_score"] for r in results}
        self.assertAlmostEqual(score_by_id[1001], expected_score_1001, places=7)
        self.assertAlmostEqual(score_by_id[1002], expected_score_1002, places=7)
        self.assertGreater(score_by_id[1001], score_by_id[1002])

class TestDiversityPass(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
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
                    "company_id": 1,
                    "final_score": 0.95,
                },
                {
                    "_point_id": 1002,
                    "score": 0.94,
                    "title": "Software Engineer",
                    "location": "Berlin",
                    "company_id": 1,
                    "final_score": 0.94,
                },
                {
                    "_point_id": 1003,
                    "score": 0.93,
                    "title": "Data Scientist",
                    "location": "Munich",
                    "company_id": 2,
                    "final_score": 0.93,
                },
                {
                    "_point_id": 1004,
                    "score": 0.92,
                    "title": "Data Scientist",
                    "location": "Munich",
                    "company_id": 2,
                    "final_score": 0.92,
                },
                {
                    "_point_id": 1005,
                    "score": 0.91,
                    "title": "DevOps Engineer",
                    "location": "Berlin",
                    "company_id": 3,
                    "final_score": 0.91,
                },
                {
                    "_point_id": 1006,
                    "score": 0.90,
                    "title": "Product Manager",
                    "location": "Hamburg",
                    "company_id": 4,
                    "final_score": 0.90,
                },
            ],
        ]

    def test_diversity_deduplication(self):

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

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_successful_ordering(self, mock_scoring_cls):

        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},  # job payload
            {"candidate_version": 1},  # candidate 101
            {"candidate_version": 2},  # candidate 102
        ]
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

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["candidate_id"], 102)
        self.assertEqual(results[0]["fit_score"], 92)
        self.assertEqual(results[0]["status"], "scored")
        self.assertEqual(results[1]["candidate_id"], 101)
        self.assertEqual(results[1]["fit_score"], 85)
        self.assertEqual(results[1]["status"], "scored")
        mock_scoring.calculate_fit.assert_any_call(101, 1, 999, 5, False)
        mock_scoring.calculate_fit.assert_any_call(102, 2, 999, 5, False)

    def test_rank_pool_missing_candidates_raises_value_error(self):

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

        self.mock_qdrant.get.return_value = None  # job missing
        with self.assertRaises(ValueError) as cm:
            self.service.rank_pool(
                job_id=999,
                job_version=5,
                candidate_ids=[101],
            )
        self.assertIn("Job not found", str(cm.exception))

    def test_rank_pool_stale_job_version_raises_value_error(self):

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
    def test_rank_pool_llm_unavailable_error_is_isolated(self, mock_scoring_cls):

        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 1},
        ]
        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.side_effect = LLMUnavailableError("circuit open")
        mock_scoring_cls.return_value = mock_scoring

        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=[101],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["candidate_id"], 101)
        self.assertEqual(results[0]["fit_score"], 0)
        self.assertEqual(results[0]["status"], "failed")

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_llm_unavailable_isolated_per_candidate(self, mock_scoring_cls):

        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 1},
            {"candidate_version": 2},
        ]
        mock_scoring = MagicMock()
        def mixed_fit(*args, **kwargs):
            if args[0] == 101:
                raise LLMUnavailableError("circuit open")
            return {"overall_score_percentage": 92}
        mock_scoring.calculate_fit.side_effect = mixed_fit
        mock_scoring_cls.return_value = mock_scoring

        results = self.service.rank_pool(
            job_id=999,
            job_version=5,
            candidate_ids=[101, 102],
        )

        self.assertEqual(len(results), 2)
        result_map = {r["candidate_id"]: r for r in results}
        self.assertEqual(result_map[101]["status"], "failed")
        self.assertEqual(result_map[101]["fit_score"], 0)
        self.assertEqual(result_map[102]["status"], "scored")
        self.assertEqual(result_map[102]["fit_score"], 92)
        self.assertGreater(result_map[102]["fit_score"], result_map[101]["fit_score"])

    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_calls_scoring_service_with_correct_versions(self, mock_scoring_cls):

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

        self.assertEqual(len(results), 100)
        result_ids = {r["candidate_id"] for r in results}
        self.assertEqual(result_ids, set(range(100)))
        for r in results:
            self.assertEqual(r["fit_score"], 80)
            self.assertEqual(r["status"], "scored")
        self.assertEqual(mock_scoring.calculate_fit.call_count, 100)

    @patch('app.services.recommendation_service.POOL_RANK_TIMEOUT_SECONDS', 1.0)
    @patch('app.services.recommendation_service.POOL_RANK_CONCURRENCY', 3)
    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_timeout_fallback(self, mock_scoring_cls):

        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
            {"candidate_version": 1},
            {"candidate_version": 2},
            {"candidate_version": 3},
        ]
        mock_scoring = MagicMock()
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

        self.assertLess(elapsed, 1.3, f"rank_pool took {elapsed:.2f}s, expected <1.3s")
        result_map = {r["candidate_id"]: r for r in results}
        self.assertEqual(result_map[101]["fit_score"], 90)
        self.assertEqual(result_map[101]["status"], "scored")
        self.assertEqual(result_map[103]["fit_score"], 70)
        self.assertEqual(result_map[103]["status"], "scored")
        self.assertEqual(result_map[102]["fit_score"], 0)
        self.assertEqual(result_map[102]["status"], "timeout")

    @patch('app.services.recommendation_service.POOL_RANK_TIMEOUT_SECONDS', 0.2)
    @patch('app.services.recommendation_service.POOL_RANK_CONCURRENCY', 3)
    @patch('app.services.recommendation_service.ScoringService')
    def test_rank_pool_timeout_scales_with_pool_size(self, mock_scoring_cls):

        self.mock_qdrant.get.side_effect = [
            {"job_version": 5},
        ] + [{"candidate_version": i} for i in range(10)]
        mock_scoring = MagicMock()
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

        self.assertEqual(len(results), 10)
        for r in results:
            self.assertEqual(r["fit_score"], 75)
            self.assertEqual(r["status"], "scored")
        self.assertLess(elapsed, 1.0)

class TestRecommendationServiceVersionValidation(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_target_version_mismatch_candidate(self):

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

class TestRecommendationCanonicalMatch(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_employment_match_canonical_case_insensitive(self):
        target = {
            "location": "Berlin",
            "experience_level": "senior",
            "employment_type": "Full-time",
            "skills": [],
        }
        result = {
            "location": "Berlin",
            "experience_level": "senior",
            "employment_type": "full-time",
            "skills": [],
        }
        score = self.service._compute_composite_score(target, result, [], 0.0, None)
        self.assertAlmostEqual(score, 0.10)

    def test_employment_match_category_mismatch_zero(self):
        target = {
            "location": "Berlin",
            "experience_level": "senior",
            "employment_type": "full-time",
            "skills": [],
        }
        result = {
            "location": "Berlin",
            "experience_level": "senior",
            "employment_type": "contract",
            "skills": [],
        }
        score = self.service._compute_composite_score(target, result, [], 0.0, None)
        self.assertAlmostEqual(score, 0.08)

    def test_location_match_uses_work_mode_resolver(self):
        target = {
            "location": "Berlin",
            "experience_level": "senior",
            "employment_type": "full-time",
            "work_mode": "remote",
            "skills": [],
        }
        result = {
            "location": "Munich",
            "experience_level": "senior",
            "employment_type": "full-time",
            "work_mode": "remote",
            "skills": [],
        }
        score = self.service._compute_composite_score(target, result, [], 0.0, None)
        self.assertAlmostEqual(score, 0.08)
class TestRecommendMinSimilarity(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _base_result(self, pid, vector_score):
        return {
            "_point_id": pid,
            "score": vector_score,
            "required_skills": ["ruby"],
            "location": "New York",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "job_version": 1,
            "title": f"Job {pid}",
        }

    def test_vector_path_prefers_above_floor(self):
        target = {
            "skills": ["python"],
            "experience_level": "mid level",
            "location": "New York",
            "employment_type": "full_time",
        }
        self.mock_qdrant.get_with_vector.return_value = (target, [0.1] * 768)
        # 0.55 scaled vector (0.50-0.90->0.0-1.0) +0.35 Jaccard +0.04 loc +0.04 level +0.02 emp
        # 1001: 0.9->scaled1.0 =>0.65 (>0.50) | 1002:0.26->0.0 =>0.10 (<0.50) | 1003:0.1->0.0 =>0.10 (<0.50)
        self.mock_qdrant.search.side_effect = [
            [],
            [
                self._base_result(1001, 0.9),
                self._base_result(1002, 0.26),
                self._base_result(1003, 0.1),
            ],
        ]
        self.mock_cache.get.return_value = None

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={},
            force_refresh=True,
            limit=10,
        )
        ids = [r["id"] for r in results]
        self.assertEqual(ids, [1001])
        self.assertAlmostEqual(results[0]["similarity_score"], 0.65, places=2)
    def test_all_below_floor_returns_empty(self):
        target = {
            "skills": ["python"],
            "experience_level": "mid level",
            "location": "New York",
            "employment_type": "full_time",
        }
        self.mock_qdrant.get_with_vector.return_value = (target, [0.1] * 768)
        self.mock_qdrant.search.side_effect = [
            [],
            [self._base_result(1001, 0.1)],  # composite 0.25 -> below hard floor 0.50 -> dropped
        ]
        self.mock_cache.get.return_value = None

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={},
            force_refresh=True,
            limit=10,
        )
        self.assertEqual(len(results), 0)
    def test_cold_start_applies_hard_floor(self):
        # Cold-start composite is metadata-only (capped at 0.40 by construction),
        # cold-start composite is metadata-only (capped at 0.50 by construction); hard floor applies.
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["python", "java"], "experience_level": "mid level", "location": "New York", "employment_type": "full_time"},
            None,
        )
        self.mock_qdrant.scroll.return_value = [
            {
                "_point_id": 1001,
                "required_skills": ["python"],
                "location": "New York",
                "experience_level": "mid level",
                "employment_type": "full_time",
                "job_version": 1,
            },
        ]
        self.mock_cache.get.return_value = None

        results = self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.assertEqual(len(results), 0)
class TestUnfilteredRetry(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.target = {
            "skills": ["python"],
            "experience_level": "mid level",
            "location": "New York",
            "employment_type": "full_time",
        }
        self.mock_qdrant.get_with_vector.return_value = (self.target, [0.1] * 768)
        self.mock_cache.get.return_value = None

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _result(self, pid, score):
        return {
            "_point_id": pid,
            "score": score,
            "required_skills": [],
            "location": "New York",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "job_version": 1,
            "title": f"Job {pid}",
        }

    def test_vector_path_filtered_empty_retries_unfiltered(self):
        # peer search, filtered main search, unfiltered retry = 3 calls
        self.mock_qdrant.search.side_effect = [
            [],
            [],
            [self._result(1001, 0.9)],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={"location": "Nowhere"},
            force_refresh=True,
            limit=10,
        )

        self.assertEqual(self.mock_qdrant.search.call_count, 3)
        retry_call = self.mock_qdrant.search.call_args_list[-1]
        self.assertIsNone(retry_call.kwargs["query_filter"])
        self.assertEqual(self.mock_qdrant.search.call_args_list[1].kwargs["query_filter"], self.service._build_filter({"location": "Nowhere"}))
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], 1001)

    def test_vector_path_no_retry_when_no_filter_set(self):
        self.mock_qdrant.search.side_effect = [[], []]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={},
            force_refresh=True,
            limit=10,
        )

        self.assertEqual(self.mock_qdrant.search.call_count, 2)
        self.assertEqual(results, [])

    def test_vector_path_no_retry_when_filtered_search_returns_results(self):
        self.mock_qdrant.search.side_effect = [
            [],
            [self._result(1001, 0.9)],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={"location": "New York"},
            force_refresh=True,
            limit=10,
        )

        self.assertEqual(self.mock_qdrant.search.call_count, 2)
        self.assertGreater(len(results), 0)

    def test_cold_start_scroll_retries_unfiltered(self):
        self.mock_qdrant.get_with_vector.return_value = (self.target, None)
        self.mock_qdrant.scroll.side_effect = [
            [],
            [dict(self._result(1001, 0.0), required_skills=["python"])],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={"location": "Nowhere"},
            force_refresh=True,
            limit=10,
        )

        self.assertEqual(self.mock_qdrant.scroll.call_count, 2)
        self.assertEqual(self.mock_qdrant.search.call_count, 0)
        self.mock_gemini.embed.assert_not_called()
        self.assertGreater(len(results), 0)

    def test_cold_start_intent_search_retries_unfiltered(self):
        self.mock_qdrant.get_with_vector.return_value = (self.target, None)
        self.mock_gemini.embed.return_value = [0.2] * 768
        self.mock_qdrant.search.side_effect = [
            [],
            [self._result(1001, 0.9)],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals={"recent_searches": ["python engineer"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={"location": "Nowhere"},
            force_refresh=True,
            limit=10,
        )

        self.assertEqual(self.mock_qdrant.search.call_count, 2)
        retry_call = self.mock_qdrant.search.call_args_list[-1]
        self.assertIsNone(retry_call.kwargs["query_filter"])
        self.mock_qdrant.scroll.assert_not_called()
        self.mock_gemini.embed.assert_called_once()
        self.assertGreater(len(results), 0)

    def test_cold_start_intent_search_uses_vector_score(self):
        self.mock_qdrant.get_with_vector.return_value = (self.target, None)
        self.mock_gemini.embed.return_value = [0.2] * 768
        self.mock_qdrant.search.side_effect = [
            [self._result(1001, 0.9)],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals={"recent_searches": ["python engineer"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=True,
            limit=10,
        )

        self.mock_qdrant.scroll.assert_not_called()
        self.mock_gemini.embed.assert_called_once()
        self.assertEqual(len(results), 1)
        # 0.9 scaled 1.0: 0.55*1 +0.10 logistics =0.65
        self.assertAlmostEqual(results[0]["similarity_score"], 0.65, places=2)
    def test_cold_start_embeds_searches_when_present(self):
        self.mock_qdrant.get_with_vector.return_value = (self.target, None)
        self.mock_gemini.embed.return_value = [0.2] * 768
        self.mock_qdrant.search.side_effect = [
            [],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals={"recent_searches": ["python engineer"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=True,
            limit=10,
        )

        self.mock_gemini.embed.assert_called_once()
        self.assertEqual(self.mock_qdrant.search.call_count, 1)
        self.mock_qdrant.scroll.assert_not_called()

    def test_cold_start_embed_failure_falls_back_to_scroll(self):
        self.mock_qdrant.get_with_vector.return_value = (self.target, None)
        self.mock_gemini.embed.side_effect = RuntimeError("embed unavailable")
        self.mock_qdrant.scroll.side_effect = [
            [dict(self._result(1001, 0.0), required_skills=["python"])],
        ]

        results = self.service.recommend(
            rec_type="jobs",
            target_id=1,
            target_version=1,
            behavioral_signals={"recent_searches": ["python engineer"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=True,
            limit=10,
        )

        self.mock_gemini.embed.assert_called_once()
        self.mock_qdrant.search.assert_not_called()
        self.assertEqual(self.mock_qdrant.scroll.call_count, 1)
        self.assertEqual(len(results), 1)

class TestSignalActivation(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )

    def test_intent_activates_with_single_search_weight(self):
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=["python"],
            recent_clicks=[],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.15)
        self.assertAlmostEqual(cooccurrence_weight, 0.0)

    def test_intent_embeds_single_search_in_vector_path(self):
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], []]
        self.mock_gemini.embed.return_value = [0.2] * 768

        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": ["python"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.mock_gemini.embed.assert_called_once()

    def test_cooc_activates_with_single_save_weight(self):
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=[],
            recent_clicks=[],
            recent_saves=[100],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.0)
        self.assertAlmostEqual(cooccurrence_weight, 0.05)

    def test_cooc_activates_with_single_click_weight(self):
        intent_weight, cooccurrence_weight, _, _ = self.service._compute_weights(
            recent_searches=[],
            recent_clicks=[{"id": 1, "dwell_time_seconds": 5}],
            recent_saves=[],
            recent_positive_outcomes=[],
        )
        self.assertAlmostEqual(intent_weight, 0.0)
        self.assertAlmostEqual(cooccurrence_weight, 0.05)

    def test_cooc_single_click_triggers_retrieval(self):
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [
            [],
            [],
        ]
        self.mock_qdrant._client = MagicMock()
        self.mock_qdrant._client.retrieve.return_value = [
            MagicMock(vector=[0.2] * 768),
        ]

        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={
                "recent_searches": [],
                "recent_clicks": [{"id": 7, "dwell_time_seconds": 5}],
                "recent_saves": [],
                "recent_positive_outcomes": [],
            },
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        retrieve_call = self.mock_qdrant._client.retrieve.call_args
        self.assertIn("ids", retrieve_call.kwargs)
        self.assertEqual(set(retrieve_call.kwargs["ids"]), {7})

    def test_cooc_single_click_not_retrieved_when_no_click_id(self):
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], []]
        self.mock_qdrant._client = MagicMock()
        self.mock_qdrant._client.retrieve.return_value = []

        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={
                "recent_searches": [],
                "recent_clicks": [],
                "recent_saves": [],
                "recent_positive_outcomes": [],
            },
            hard_filters={},
            force_refresh=False,
            limit=10,
        )
        self.mock_qdrant._client.retrieve.assert_not_called()

    def test_single_signal_weights_sum_to_one(self):
        signal_sets = [
            (["python"], [], [], []),
            ([], [{"id": 1, "dwell_time_seconds": 5}], [], []),
            ([], [], [100], []),
        ]
        for searches, clicks, saves, pos in signal_sets:
            iw, cw, pw, ppw = self.service._compute_weights(searches, clicks, saves, pos)
            self.assertAlmostEqual(iw + cw + pw + ppw, 1.0, places=10)

class TestRecallCeiling(unittest.TestCase):

    def setUp(self):
        self.mock_gemini = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.service = RecommendationService(
            llm=self.mock_gemini,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
        )
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], []]

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def test_vector_search_fetches_effective_limit_times_five(self):
        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals=self._signals(),
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        main_search_call = self.mock_qdrant.search.call_args_list[-1]
        self.assertEqual(main_search_call.kwargs["limit"], 50)

    def test_cold_start_intent_search_fetches_effective_limit_times_five(self):
        self.mock_qdrant.get_with_vector.return_value = (
            {"skills": ["a", "b", "c"]}, None,
        )
        self.mock_gemini.embed.return_value = [0.2] * 768
        self.mock_qdrant.search.side_effect = [[]]

        self.service.recommend(
            rec_type="jobs",
            target_id=123,
            target_version=1,
            behavioral_signals={"recent_searches": ["python"], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []},
            hard_filters={},
            force_refresh=False,
            limit=10,
        )

        main_search_call = self.mock_qdrant.search.call_args_list[-1]
        self.assertEqual(main_search_call.kwargs["limit"], 50)

if __name__ == "__main__":
    unittest.main()
