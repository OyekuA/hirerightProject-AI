import json
import unittest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from app.services.career_service import CareerPathService, MalformedLLMResponseError
from app.clients.llm import LLMUnavailableError
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

    def setUp(self):

        self.gemini_mock = MagicMock()
        self.qdrant_mock = MagicMock()
        self.cache_mock = MagicMock()
        self.cache_mock.get.return_value = None  # cache miss by default
        self.service = CareerPathService(
            llm=self.gemini_mock,
            qdrant=self.qdrant_mock,
            cache=self.cache_mock,
        )
        self.truncate_patcher = patch(
            "app.services.career_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()
        self.settings_patcher = patch("app.services.career_service.get_settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value.LLM_SEED = 42
        self.mock_settings.return_value.CACHE_TTL_SECONDS = 86400

    def tearDown(self):
        self.truncate_patcher.stop()
        self.settings_patcher.stop()

    def test_happy_path_returns_three_items(self):

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

        self.qdrant_mock.get.return_value = None
        with self.assertRaises(ValueError) as cm:
            self.service.analyze_career_paths(candidate_id=999)
        self.assertIn("not found", str(cm.exception).lower())
        self.gemini_mock.generate.assert_not_called()

    def test_llm_non_json_raises_error(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(MalformedLLMResponseError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.assertIn("malformed", str(cm.exception).lower())
        self.assertEqual(self.gemini_mock.generate.call_count, 2)  # original + retry

    def test_core_skills_not_a_list_succeeds(self):

        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = "Python"
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(result["paths"][0]["core_skills"], [])
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_empty_list_succeeds(self):

        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = []
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(result["paths"][0]["core_skills"], [])
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_non_string_item_succeeds(self):

        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["Python", 123, "SQL"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(result["paths"][0]["core_skills"], ["Python", "SQL"])
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_too_few_succeeds(self):

        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["Python", "SQL"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(result["paths"][0]["core_skills"], ["Python", "SQL"])
        self.gemini_mock.generate.assert_called_once()

    def test_core_skills_too_many_returns_data_as_is(self):

        mutated = VALID_ITEM.copy()
        mutated["core_skills"] = ["a", "b", "c", "d", "e", "f"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(len(result["paths"][0]["core_skills"]), 6)
        self.gemini_mock.generate.assert_called_once()

    def test_profile_summary_missing_raises_error(self):

        response = {
            "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]
        }
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(MalformedLLMResponseError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_profile_summary_empty_raises_error(self):

        response = {
            "profile_summary": "",
            "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]
        }
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps(response)
        with self.assertRaises(MalformedLLMResponseError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_reasoning_missing_raises_error(self):

        mutated = VALID_ITEM.copy()
        del mutated["reasoning"]
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(MalformedLLMResponseError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_reasoning_empty_raises_error(self):

        mutated = VALID_ITEM.copy()
        mutated["reasoning"] = ""
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = json.dumps({
            "profile_summary": VALID_PROFILE_SUMMARY,
            "paths": [mutated, VALID_ITEM, VALID_ITEM]
        })
        with self.assertRaises(MalformedLLMResponseError) as cm:
            self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()

    def test_prompt_contains_second_person_instruction(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("second", prompt.lower())
        self.assertIn("you", prompt.lower())
        self.assertIn("your", prompt.lower())

    def test_prompt_contains_core_skills_description(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("core_skills", prompt)

    def test_prompt_contains_profile_summary_description(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("profile_summary", prompt)

    def test_prompt_contains_grounding_constraint(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("GROUNDING CONSTRAINT", prompt)

    def test_prompt_contains_bias_guard(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("BIAS GUARD", prompt)

    def test_prompt_does_not_contain_tech_only_examples(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertNotIn("Senior Data Engineer", prompt)
        self.assertNotIn("ETL pipelines", prompt)
        self.assertIn("HR Operations Manager", prompt)

    def test_name_is_not_sent_to_llm(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"], "location": "Remote, US"}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertNotIn("Alice", prompt)

    def test_uses_low_temperature_and_seed(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"]}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        call_kwargs = self.gemini_mock.generate.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.2)
        self.assertEqual(call_kwargs["seed"], 42)
        self.assertIn("response_format", call_kwargs)

    def test_cache_hit_skips_llm_call(self):

        cached_result = {"profile_summary": "cached", "paths": [VALID_ITEM, VALID_ITEM, VALID_ITEM]}
        self.cache_mock.get.return_value = cached_result
        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"], "candidate_version": 1}
        result = self.service.analyze_career_paths(candidate_id=42)
        self.assertEqual(result, cached_result)
        self.gemini_mock.generate.assert_not_called()
        self.cache_mock.get.assert_called_once_with("42:1:career")

    def test_cache_miss_stores_result(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"], "candidate_version": 1}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        result = self.service.analyze_career_paths(candidate_id=42)
        self.gemini_mock.generate.assert_called_once()
        self.cache_mock.set.assert_called_once()
        cache_key = self.cache_mock.set.call_args[0][0]
        self.assertEqual(cache_key, "42:1:career")

    def test_cache_key_uses_candidate_version(self):

        self.qdrant_mock.get.return_value = {"name": "Alice", "skills": ["Python"], "candidate_version": 3}
        self.gemini_mock.generate.return_value = VALID_RESPONSE
        self.service.analyze_career_paths(candidate_id=42)
        self.cache_mock.set.assert_called_once()
        cache_key = self.cache_mock.set.call_args[0][0]
        self.assertEqual(cache_key, "42:3:career")
