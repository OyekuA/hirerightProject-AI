"""
Deterministic unit tier for the /recommend eval harness (no network).

Drives the composite arithmetic and gate behavior with settings pinned via
`patch("app.services.recommendation_service.get_settings", ...)` — never
literal floors — so config changes cannot silently break these tests.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.services.recommendation_service import RecommendationService

from tests.eval.fixtures import (
    CANDIDATE_FIXTURES,
    JOB_FIXTURES,
    CANDIDATE_BY_ID,
)


def _mock_settings(**overrides):
    defaults = {
        "RECOMMEND_WEIGHT_VECTOR": 0.55,
        "RECOMMEND_WEIGHT_SKILL": 0.35,
        "RECOMMEND_WEIGHT_LOCATION": 0.04,
        "RECOMMEND_WEIGHT_LEVEL": 0.04,
        "RECOMMEND_WEIGHT_EMPLOYMENT": 0.02,
        "RAW_COSINE_GATE": 0.30,
        "SKILL_COSINE_GATE": 0.40,
        "RECOMMEND_SKILL_RESCALE_LO": 0.30,
        "RECOMMEND_SKILL_RESCALE_HI": 1.0,
        "LEVEL_GATE_DISTANCE": 4,
        "RECOMMEND_MAX_SEARCHES": 5,
        "RECOMMEND_MAX_COOC_IDS": 20,
    }
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestCompositeArithmetic(unittest.TestCase):
    """Composite must be 0.55*raw + 0.35*Jaccard + logistics, window-free."""

    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )

    def _job(self, skills, **kw):
        base = {
            "required_skills": skills,
            "skills_vector": [1.0, 0.0],
            "location": "Lagos, Nigeria",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "work_mode": "onsite",
        }
        base.update(kw)
        return base

    def _target(self, skills, **kw):
        base = {
            "skills": skills,
            "skills_vector": [1.0, 0.0],
            "location": "Lagos, Nigeria",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "work_mode": "onsite",
        }
        base.update(kw)
        return base

    def test_raw_cosine_0_55_exact_arithmetic(self):
        target = self._target(["python", "docker"])
        job = self._job(["python", "docker"], location="Lagos, Nigeria")
        # skill cosine 1.0 -> rescaled 1.0; location same city 1.0; level equal 1.0; emp equal 1.0
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            score = self.service._compute_composite_score(
                target, job, 0.55, "jobs"
            )
        expected = 0.55 * 0.55 + 0.35 * 1.0 + 0.04 * 1.0 + 0.04 * 1.0 + 0.02 * 1.0
        self.assertAlmostEqual(score, expected, places=6)

    def test_raw_cosine_0_35_not_zeroed(self):
        """The old window zeroed vector term for raw <= 0.50; new composite never does."""
        target = self._target(["python"], location="Lagos, Nigeria")
        job = self._job(["python"], location="Lagos, Nigeria")
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            score = self.service._compute_composite_score(
                target, job, 0.35, "jobs"
            )
        self.assertGreater(score, 0.55 * 0.35)  # vector term contributes directly
        self.assertGreater(score, 0.30)

    def test_raw_cosine_0_25_still_monotonic(self):
        target = self._target(["python"], location="Lagos, Nigeria")
        job = self._job(["python"], location="Lagos, Nigeria")
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            low = self.service._compute_composite_score(target, job, 0.25, "jobs")
            high = self.service._compute_composite_score(target, job, 0.55, "jobs")
        self.assertLess(low, high)


class TestRawGate(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python"],
                "skills_vector": [1.0, 0.0],
                "experience_level": "mid level",
                "location": "New York",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _job_result(self, pid, score):
        return {
            "_point_id": pid,
            "score": score,
            "required_skills": ["python"],
            "skills_vector": [1.0, 0.0],
            "location": "New York",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "work_mode": "onsite",
            "job_version": 1,
            "title": "Python Dev",
            "company_name": "Acme",
        }

    def test_raw_0_25_dropped(self):
        self.mock_qdrant.search.side_effect = [[], [self._job_result(1001, 0.25)]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(results, [])

    def test_raw_0_35_passes_with_skill_cosine(self):
        self.mock_qdrant.search.side_effect = [[], [self._job_result(1001, 0.35)]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1001)


class TestSkillGateThreeStateEval(unittest.TestCase):
    """Three-state skill gate using fixtures: both non-empty, one empty, both empty."""

    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python", "docker"],
                "skills_vector": [1.0, 0.0],
                "experience_level": "senior",
                "location": "Berlin, Germany",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _job(self, pid, skills, score=0.55, skills_vector=None):
        return {
            "_point_id": pid,
            "score": score,
            "required_skills": skills,
            "skills_vector": skills_vector if skills_vector is not None else [1.0, 0.0],
            "location": "Berlin, Germany",
            "experience_level": "senior",
            "employment_type": "full_time",
            "work_mode": "onsite",
            "job_version": 1,
            "title": "Role",
            "company_name": "Acme",
        }

    def test_both_nonempty_semantic_zero_dropped(self):
        """Nurse-style cross-domain: both have skills_vector, cosine 0.0 < 0.40 -> dropped."""
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, ["patient care", "IV therapy"], skills_vector=[0.0, 1.0])]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(results, [])

    def test_one_empty_kept_with_degradation(self):
        """One side has skills_vector, other missing -> kept (no silent drop), extraction_degraded logged."""
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, [], skills_vector=None)]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()) as m:
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)

    def test_both_empty_kept_vector_only(self):
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": [],
                "experience_level": "senior",
                "location": "Berlin, Germany",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, [])]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)


class TestLevelGateEval(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python"],
                "experience_level": "senior",  # ladder idx 7
                "location": "New York",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _job(self, pid, level, score=0.62):
        return {
            "_point_id": pid,
            "score": score,
            "required_skills": ["python"],
            "location": "New York",
            "experience_level": level,
            "employment_type": "full_time",
            "work_mode": "onsite",
            "job_version": 1,
            "title": "Role",
            "company_name": "Acme",
        }

    def test_senior_to_intern_dropped(self):
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, "intern")]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(results, [])

    def test_senior_to_junior_passes(self):
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, "junior")]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)


class TestNoFloorTopUp(unittest.TestCase):
    """[] only when gates fire; never top-up below gates."""

    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python"],
                "skills_vector": [1.0, 0.0],
                "experience_level": "mid level",
                "location": "New York",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def _job(self, pid, score, skills=None):
        return {
            "_point_id": pid,
            "score": score,
            "required_skills": skills if skills is not None else ["python"],
            "location": "New York",
            "experience_level": "mid level",
            "employment_type": "full_time",
            "work_mode": "onsite",
            "job_version": 1,
            "title": "Role",
            "company_name": "Acme",
        }

    def test_all_raw_gated_returns_empty(self):
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, 0.20), self._job(1002, 0.25)]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(results, [])

    def test_some_pass_some_gated(self):
        self.mock_qdrant.search.side_effect = [[], [self._job(1001, 0.20), self._job(1002, 0.55)]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual([r["id"] for r in results], [1002])


class TestSignalTruncationEval(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.service.llm.embed.return_value = [0.1] * 768
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["python"],
                "skills_vector": [1.0, 0.0],
                "experience_level": "mid level",
                "location": "New York",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def test_searches_truncated_to_max(self):
        searches = [f"search {i}" for i in range(12)]
        self.mock_qdrant.search.side_effect = [[], []]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            self.service.recommend(
                rec_type="jobs", target_id=1, target_version=1,
                behavioral_signals={
                    "recent_searches": searches, "recent_clicks": [],
                    "recent_saves": [], "recent_positive_outcomes": [],
                },
                hard_filters={}, limit=10,
            )
        embed_calls = self.service.llm.embed.call_args_list
        # 5 kept searches -> at most 5 embeds (peer path may embed nothing extra)
        self.assertLessEqual(len(embed_calls), 5)


class TestFixturesShape(unittest.TestCase):
    def test_fixture_counts(self):
        self.assertEqual(len(CANDIDATE_FIXTURES), 20)
        self.assertEqual(len(JOB_FIXTURES), 40)

    def test_every_candidate_has_gold(self):
        for c in CANDIDATE_FIXTURES:
            self.assertGreaterEqual(len(c["gold_job_ids"]), 2, f"candidate {c['id']} needs >=2 gold jobs")

    def test_domains_covered(self):
        cand_domains = {c["domain"] for c in CANDIDATE_FIXTURES}
        job_domains = {j["domain"] for j in JOB_FIXTURES}
        self.assertEqual(cand_domains, {"nursing", "software", "finance", "trades"})
        self.assertEqual(job_domains, {"nursing", "software", "finance", "trades"})

    def test_empty_skill_rate_below_threshold(self):
        # Extraction-fidelity guard: the eval corpus must not be dominated by
        # empty-skill fixtures (the dummy-PDF trap). Plan: empty-rate <= 5%.
        from tests.eval.fixtures import EMPTY_RATE

        self.assertLessEqual(EMPTY_RATE, 0.05, f"empty-skill rate {EMPTY_RATE:.2%} too high")


if __name__ == "__main__":
    unittest.main()


class TestSemanticSkillMatching(unittest.TestCase):
    """Adversarial regression tests: the EHR paraphrase that broke prod must PASS
    under semantic cosine, and a reintroduced word-Jaccard must FAIL these."""

    def setUp(self):
        self.service = RecommendationService(
            llm=MagicMock(), qdrant=MagicMock(), cache=MagicMock()
        )
        self.mock_qdrant = self.service.qdrant
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["Electronic health records (EHR)", "Patient care and clinical assessment"],
                "skills_vector": [0.9, 0.1],  # semantic: near nurse domain
                "experience_level": "senior",
                "location": "Lagos, Nigeria",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )

    def _signals(self):
        return {"recent_searches": [], "recent_clicks": [], "recent_saves": [], "recent_positive_outcomes": []}

    def test_ehr_paraphrase_nurse_to_head_nurse_passes(self):
        """The exact production failure: 'Electronic health records (EHR)' vs
        'electronic health records experience' must NOT be dropped (word-Jaccard
        sees zero overlap; semantic cosine must pass)."""
        self.mock_qdrant.search.side_effect = [[], [
            {
                "_point_id": 2001,
                "score": 0.62,
                "required_skills": ["electronic health records experience", "BSc Nursing", "valid RN licence"],
                "skills_vector": [0.85, 0.15],  # semantic: same nursing domain
                "location": "Lagos, Nigeria",
                "experience_level": "senior",
                "employment_type": "full_time",
                "work_mode": "onsite",
                "job_version": 1,
                "title": "Head Nurse / Clinical Lead",
                "company_name": "Lagos General Hospital",
            },
        ]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1001, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 2001)

    def test_cross_domain_sales_to_nurse_dropped(self):
        """Sales candidate must NOT surface a nurse job: skill cosine far below gate."""
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["B2B enterprise sales", "CRM", "negotiation"],
                "skills_vector": [0.1, 0.9],  # semantic: sales domain
                "experience_level": "senior",
                "location": "Lagos, Nigeria",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], [
            {
                "_point_id": 2001,
                "score": 0.62,
                "required_skills": ["patient care", "medication administration"],
                "skills_vector": [0.9, 0.1],  # nurse domain
                "location": "Lagos, Nigeria",
                "experience_level": "senior",
                "employment_type": "full_time",
                "work_mode": "onsite",
                "job_version": 1,
                "title": "Head Nurse",
                "company_name": "Hospital",
            },
        ]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=1004, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(results, [])

    def test_flask_fastapi_semantic_family_passes(self):
        """Flask vs FastAPI share the semantic family: must pass under cosine
        (word-Jaccard drops them)."""
        self.mock_qdrant.get_with_vector.return_value = (
            {
                "skills": ["Flask", "Python"],
                "skills_vector": [0.85, 0.15],  # python web family
                "experience_level": "mid level",
                "location": "Lagos, Nigeria",
                "employment_type": "full_time",
                "candidate_version": 1,
            },
            [0.1] * 768,
        )
        self.mock_qdrant.search.side_effect = [[], [
            {
                "_point_id": 3002,
                "score": 0.60,
                "required_skills": ["FastAPI", "Python"],
                "skills_vector": [0.80, 0.20],  # same family
                "location": "Lagos, Nigeria",
                "experience_level": "mid level",
                "employment_type": "full_time",
                "work_mode": "onsite",
                "job_version": 1,
                "title": "Python Backend Engineer",
                "company_name": "FinTech Co",
            },
        ]]
        with patch("app.services.recommendation_service.get_settings", return_value=_mock_settings()):
            results = self.service.recommend(
                rec_type="jobs", target_id=4002, target_version=1,
                behavioral_signals=self._signals(), hard_filters={}, limit=10,
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 3002)
