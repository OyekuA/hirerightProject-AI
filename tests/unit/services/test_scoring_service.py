

import json
import unittest
from unittest.mock import MagicMock, patch
from app.services.scoring_service import ScoringService
from app.clients.llm import LLMUnavailableError

LLM_RESPONSE = {
    "skills": {"score": 85, "short_reason": "Strong skill overlap; Go experience credited toward Python requirement."},
    "role_match": {"score": 75, "short_reason": "Related domain with relevant past roles."},
    "skill_gap_analysis": "Candidate lacks Kubernetes experience.",
}

VALID_RESULT = {
    "overall_score_percentage": 69,
    "category_breakdown": {
        "skills": {"score": 85, "status": "pass", "short_reason": "Strong skill overlap; Go experience credited toward Python requirement."},
        "role_match": {"score": 75, "status": "pass", "short_reason": "Related domain with relevant past roles."},
        "experience": {"score": 50, "status": "warning", "short_reason": "Insufficient experience level data; neutral score assigned."},
        "location": {"score": 50, "status": "warning", "short_reason": "Insufficient location data; neutral score assigned."},
        "employment_type": {"score": 50, "status": "warning", "short_reason": "Insufficient employment type data; neutral score assigned."},
    },
    "skill_gap_analysis": "Candidate lacks Kubernetes experience.",
}

CANDIDATE_PAYLOAD = {
    "candidate_version": 2,
    "name": "Alice",
    "experience_level": "senior",
    "location": "Berlin, Germany",
    "employment_type": "full-time",
    "skills": ["Python", "Go", "Docker"],
}

JOB_PAYLOAD = {
    "job_version": 4,
    "title": "Software Engineer",
    "experience_level": "senior",
    "location": "Berlin, Germany",
    "employment_type": "full-time",
    "required_skills": ["Python", "Kubernetes"],
}

class TestScoringService(unittest.TestCase):

    def setUp(self):

        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.cache_mock = MagicMock()
        self.service = ScoringService(
            llm=self.gemini_mock,
            qdrant=self.qdrant_mock,
            cache=self.cache_mock,
        )
        settings_mock = MagicMock(
            CACHE_TTL_SECONDS=86400,
            MAX_PROMPT_CHARS=50000,
            SCORING_WEIGHT_SKILLS=0.35,
            SCORING_WEIGHT_ROLE=0.25,
            SCORING_WEIGHT_EXPERIENCE=0.20,
            SCORING_WEIGHT_LOCATION=0.12,
            SCORING_WEIGHT_EMPLOYMENT=0.08,
            SCORING_STATUS_PASS_THRESHOLD=75,
            SCORING_STATUS_WARNING_THRESHOLD=50,
        )
        self.settings_patcher = patch(
            "app.services.scoring_service.get_settings",
            return_value=settings_mock,
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result["overall_score_percentage"], 88)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once()

    def test_force_refresh_bypasses_cache(self):

        self.cache_mock.get.return_value = VALID_RESULT
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=True,
        )
        self.assertEqual(result["overall_score_percentage"], 88)
        self.cache_mock.get.assert_not_called()
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once()

    def test_candidate_not_found_raises_value_error(self):

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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps({"role_match": {}, "skill_gap_analysis": "gap"})
        with self.assertRaises(LLMUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_llm_score_out_of_range_raises_error(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        bad = {
            "skills": {"score": 150, "short_reason": "Too high"},
            "role_match": {"score": 75, "short_reason": "Good"},
            "skill_gap_analysis": "gap",
        }
        self.gemini_mock.generate.return_value = json.dumps(bad)
        with self.assertRaises(LLMUnavailableError):
            self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_llm_score_non_integer_raises_error(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        bad = {
            "skills": {"score": "eighty-five", "short_reason": "Bad"},
            "role_match": {"score": 75, "short_reason": "Good"},
            "skill_gap_analysis": "gap",
        }
        self.gemini_mock.generate.return_value = json.dumps(bad)
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result["overall_score_percentage"], 88)
        self.cache_mock.get.assert_called_once_with("1:2:3:4")
        self.assertEqual(self.qdrant_mock.get.call_count, 2)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        self.cache_mock.set.assert_called_once()

    def test_cache_key_format(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.cache_mock.get.assert_called_once_with("1:2:3:4")
        self.cache_mock.set.assert_called_once()

    def test_prompt_contains_no_pronoun_rule(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
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

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        llm_data = {
            "skills": {"score": 80, "short_reason": "Good skills fit"},
            "role_match": {"score": 70, "short_reason": "Relevant domain"},
            "skill_gap_analysis": "Some gaps",
        }
        self.gemini_mock.generate.return_value = json.dumps(llm_data)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertEqual(result["overall_score_percentage"], 86)
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_calculate_fit_malformed_json_raises_error(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
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

    def test_calculate_fit_prompt_contains_skills_and_role_match(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("**skills**", prompt)
        self.assertIn("**role_match**", prompt)
        self.assertNotIn("SCORING RUBRIC", prompt)
        self.assertNotIn("Tier 2", prompt)

    def test_score_from_payloads_aggregation_math(self):

        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=CANDIDATE_PAYLOAD,
            job_payload=JOB_PAYLOAD,
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
        )
        self.assertEqual(result["overall_score_percentage"], 88)
        self.assertIn("skills", result["category_breakdown"])
        self.assertIn("role_match", result["category_breakdown"])
        self.assertIn("experience", result["category_breakdown"])
        self.assertIn("location", result["category_breakdown"])
        self.assertIn("employment_type", result["category_breakdown"])

    def test_score_from_payloads_missing_experience_fallback(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand.pop("experience_level", None)
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=JOB_PAYLOAD,
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
        )
        exp = result["category_breakdown"]["experience"]
        self.assertEqual(exp["score"], 50)
        self.assertEqual(exp["status"], "warning")
        self.assertIn("Insufficient", exp["short_reason"])

    def test_score_from_payloads_missing_location_fallback(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand.pop("location", None)
        job = dict(JOB_PAYLOAD)
        job.pop("location", None)
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
        )
        loc = result["category_breakdown"]["location"]
        self.assertEqual(loc["score"], 50)
        self.assertEqual(loc["status"], "warning")
        self.assertIn("Insufficient location data", loc["short_reason"])

    def test_score_from_payloads_missing_employment_type_fallback(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand.pop("employment_type", None)
        job = dict(JOB_PAYLOAD)
        job.pop("employment_type", None)
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
        )
        emp = result["category_breakdown"]["employment_type"]
        self.assertEqual(emp["score"], 50)
        self.assertEqual(emp["status"], "warning")
        self.assertIn("Insufficient employment type data", emp["short_reason"])

    def test_score_from_payloads_is_delegated_by_calculate_fit(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        with patch.object(self.service, 'score_from_payloads', wraps=self.service.score_from_payloads) as spy:
            result = self.service.calculate_fit(
                candidate_id=1,
                candidate_version=2,
                job_id=3,
                job_version=4,
                force_refresh=False,
            )
            spy.assert_called_once()
            self.assertIn("overall_score_percentage", result)
            self.assertIn("category_breakdown", result)
            self.assertIn("skill_gap_analysis", result)

    def test_legacy_cache_format_recomputed(self):

        old_format = {
            "overall_score_percentage": 75,
            "category_breakdown": {
                "role_match": {"status": "pass", "short_reason": "Good fit"},
                "experience": {"status": "pass", "short_reason": "Relevant"},
                "location": {"status": "pass", "short_reason": "Same city"},
                "employment_type": {"status": "pass", "short_reason": "Full-time"},
            },
            "skill_gap_analysis": "Some gaps",
        }
        self.cache_mock.get.return_value = old_format
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertIn("skills", result.get("category_breakdown", {}))
        self.assertEqual(self.gemini_mock.generate.call_count, 1)
        for dim in ("skills", "role_match", "experience", "location", "employment_type"):
            self.assertIn("score", result["category_breakdown"][dim])

    def test_legacy_cache_missing_score_field_recomputed(self):

        old_format = {
            "overall_score_percentage": 75,
            "category_breakdown": {
                "skills": {"status": "pass", "short_reason": "Good"},
                "role_match": {"status": "pass", "short_reason": "Fit"},
                "experience": {"status": "pass", "short_reason": "Exp"},
                "location": {"status": "pass", "short_reason": "Loc"},
                "employment_type": {"status": "pass", "short_reason": "Emp"},
            },
            "skill_gap_analysis": "Gap",
        }
        self.cache_mock.get.return_value = old_format
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        self.assertIn("score", result["category_breakdown"]["skills"])
        self.assertEqual(self.gemini_mock.generate.call_count, 1)

    def test_remote_job_missing_location_passes(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "remote"
        cand["location"] = ""  # missing location
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "remote"
        job["location"] = ""  # missing location
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        loc = result["category_breakdown"]["location"]
        self.assertEqual(loc["score"], 100, msg=f"Expected pass for remote match, got {loc}")
        self.assertEqual(loc["status"], "pass")

    def test_remote_job_candidate_not_open_to_remote_without_location(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "full-time"  # not remote
        cand["location"] = ""
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "remote"
        job["location"] = ""
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        loc = result["category_breakdown"]["location"]
        self.assertEqual(loc["score"], 50)
        self.assertEqual(loc["status"], "warning")
        self.assertIn("Insufficient location data", loc["short_reason"])

    def test_hybrid_job_missing_location_passes(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "hybrid"
        cand["location"] = ""
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "hybrid"
        job["location"] = ""
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        loc = result["category_breakdown"]["location"]
        self.assertEqual(loc["score"], 100, msg=f"Expected pass for hybrid match, got {loc}")
        self.assertEqual(loc["status"], "pass")

    def test_employment_type_contract_vs_fulltime_fails(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "full-time"
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "contract"
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        emp = result["category_breakdown"]["employment_type"]
        self.assertEqual(emp["score"], 20)
        self.assertEqual(emp["status"], "fail")

    def test_employment_type_parttime_vs_internship_fails(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "part-time"
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "internship"
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        emp = result["category_breakdown"]["employment_type"]
        self.assertEqual(emp["score"], 20)
        self.assertEqual(emp["status"], "fail")

    def test_employment_type_same_category_diff_arrangement_warning(self):

        cand = dict(CANDIDATE_PAYLOAD)
        cand["employment_type"] = "full-time remote"
        job = dict(JOB_PAYLOAD)
        job["employment_type"] = "full-time"
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        result = self.service.score_from_payloads(
            candidate_payload=cand,
            job_payload=job,
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        emp = result["category_breakdown"]["employment_type"]
        self.assertEqual(emp["score"], 60)
        self.assertEqual(emp["status"], "warning")

    def test_bias_masking_removes_name_from_prompt(self):

        self.cache_mock.get.return_value = None
        self.qdrant_mock.get.side_effect = [
            dict(CANDIDATE_PAYLOAD),
            dict(JOB_PAYLOAD),
        ]
        self.gemini_mock.generate.return_value = json.dumps(LLM_RESPONSE)
        self.service.calculate_fit(
            candidate_id=1,
            candidate_version=2,
            job_id=3,
            job_version=4,
            force_refresh=False,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]

        self.assertNotIn("Alice", prompt)
        self.assertIn("Germany", prompt)

        result = self.service.score_from_payloads(
            candidate_payload=dict(CANDIDATE_PAYLOAD),
            job_payload=dict(JOB_PAYLOAD),
            candidate_id=1, candidate_version=2,
            job_id=3, job_version=4,
        )
        loc = result["category_breakdown"]["location"]
        self.assertEqual(loc["score"], 100)
        self.assertEqual(loc["status"], "pass")
