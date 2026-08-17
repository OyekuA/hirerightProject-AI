import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from app.clients.llm import LLMUnavailableError


class TestDecisionService(IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()

    def _make_service(self):
        from app.services.decision_service import DecisionService
        return DecisionService(llm=self.mock_llm, qdrant=self.mock_qdrant, cache=self.mock_cache)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_hire_when_both_scores_high(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 85,
            "category_breakdown": {"skills": {"score": 90, "status": "pass", "short_reason": "ok"}},
            "skill_gap_analysis": "None.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "Strong.", "confidence": 92})

        service = self._make_service()
        result = service.decide(candidate_id=1, candidate_version=1, job_id=10, job_version=1, assessment_score=82)

        self.assertEqual(result["decision"], "hire")
        self.assertEqual(result["rationale"], "Strong.")
        self.assertEqual(result["confidence"], 92)
        self.assertEqual(result["combined_score"], round(0.4 * 85 + 0.6 * 82))
        self.assertEqual(self.mock_llm.generate.call_count, 1)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_no_hire_when_both_scores_low(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 30,
            "category_breakdown": {"skills": {"score": 25, "status": "fail", "short_reason": "poor"}},
            "skill_gap_analysis": "Major gaps.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "Weak.", "confidence": 95})

        service = self._make_service()
        result = service.decide(candidate_id=2, candidate_version=1, job_id=20, job_version=1, assessment_score=25)

        self.assertEqual(result["decision"], "no_hire")
        self.assertEqual(result["confidence"], 95)
        self.assertEqual(self.mock_llm.generate.call_count, 1)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_review_when_borderline(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 55,
            "category_breakdown": {"skills": {"score": 60, "status": "warning", "short_reason": "ok"}},
            "skill_gap_analysis": "Some gaps.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "Borderline.", "confidence": 55})

        service = self._make_service()
        result = service.decide(candidate_id=3, candidate_version=1, job_id=30, job_version=1, assessment_score=50)

        self.assertEqual(result["decision"], "review")
        self.assertEqual(self.mock_llm.generate.call_count, 1)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_fusion_math_correctness(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 80,
            "category_breakdown": {"skills": {"score": 80, "status": "pass", "short_reason": "ok"}},
            "skill_gap_analysis": "None.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "OK.", "confidence": 80})

        service = self._make_service()
        result = service.decide(candidate_id=1, candidate_version=1, job_id=10, job_version=1, assessment_score=90)

        self.assertEqual(result["combined_score"], 86)   # round(0.4*80 + 0.6*90) = round(32+54) = 86
        self.assertEqual(result["fit_score"], 80)
        self.assertEqual(result["assessment_score"], 90)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_malformed_llm_response_raises(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 80,
            "category_breakdown": {"skills": {"score": 80, "status": "pass", "short_reason": "ok"}},
            "skill_gap_analysis": "None.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = "not valid json"

        service = self._make_service()
        with self.assertRaises(LLMUnavailableError):
            service.decide(candidate_id=1, candidate_version=1, job_id=10, job_version=1, assessment_score=80)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_needs_review_forces_review_label(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60
        mock_settings.return_value.LLM_SEED = 7

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 85,
            "category_breakdown": {"skills": {"score": 90, "status": "pass", "short_reason": "ok"}},
            "skill_gap_analysis": "None.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "Flagged for review.", "confidence": 80})

        service = self._make_service()
        result = service.decide(candidate_id=1, candidate_version=1, job_id=10, job_version=1, assessment_score=82, needs_review=True)

        self.assertEqual(result["decision"], "review")
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertIn("needs_review", prompt)
        self.assertIn("forced to review", prompt)
        self.assertEqual(self.mock_llm.generate.call_args.kwargs.get("seed"), 7)

    @patch("app.services.decision_service.ScoringService")
    @patch("app.services.decision_service.get_settings")
    def test_high_scores_still_hire_without_needs_review(self, mock_settings, mock_scoring_cls):
        mock_settings.return_value.DECISION_FIT_WEIGHT = 0.40
        mock_settings.return_value.DECISION_ASSESSMENT_WEIGHT = 0.60
        mock_settings.return_value.LLM_SEED = 7

        mock_scoring = MagicMock()
        mock_scoring.calculate_fit.return_value = {
            "overall_score_percentage": 85,
            "category_breakdown": {"skills": {"score": 90, "status": "pass", "short_reason": "ok"}},
            "skill_gap_analysis": "None.",
        }
        mock_scoring_cls.return_value = mock_scoring

        self.mock_llm.generate.return_value = json.dumps({"rationale": "Strong.", "confidence": 92})

        service = self._make_service()
        result = service.decide(candidate_id=1, candidate_version=1, job_id=10, job_version=1, assessment_score=82, needs_review=False)

        self.assertEqual(result["decision"], "hire")
        prompt = self.mock_llm.generate.call_args[0][0]
        self.assertNotIn("needs_review", prompt)
