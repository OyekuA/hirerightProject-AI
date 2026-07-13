

import unittest
from unittest.mock import MagicMock, patch
from app.services.jd_service import JDService
from app.clients.llm import LLMUnavailableError
from app.clients.qdrant import QdrantClient, JOBS_COLLECTION

class TestJDService(unittest.TestCase):

    def setUp(self):

        self.gemini_mock = MagicMock()
        self.service = JDService(llm=self.gemini_mock)
        self.truncate_patcher = patch(
            "app.services.jd_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.truncate_patcher.stop()

    def test_generate_jd_fresh_returns_string(self):

        self.gemini_mock.generate.return_value = "Job Title: Senior Engineer\nResponsibilities: ..."
        result = self.service.generate_jd(
            prompt="We need a senior backend engineer with Python and Kubernetes experience.",
            existing_draft=None,
        )
        self.assertEqual(result, "Job Title: Senior Engineer\nResponsibilities: ...")
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_fresh_prompt_forbids_markdown(self):

        self.gemini_mock.generate.return_value = "Job Title: Engineer"
        self.service.generate_jd(
            prompt="Some prompt",
            existing_draft=None,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Bold", prompt)
        self.assertIn("##", prompt)

    def test_generate_jd_refinement_returns_string(self):

        self.gemini_mock.generate.return_value = "Job Title: Senior Engineer\nResponsibilities: ..."
        result = self.service.generate_jd(
            prompt="Make it more detailed.",
            existing_draft="Some draft",
        )
        self.assertEqual(result, "Job Title: Senior Engineer\nResponsibilities: ...")
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_refinement_prompt_forbids_markdown(self):

        self.gemini_mock.generate.return_value = "Job Title: Engineer"
        self.service.generate_jd(
            prompt="Some prompt",
            existing_draft="Some draft",
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Bold", prompt)
        self.assertIn("##", prompt)

    def test_generate_jd_llm_exception_raises_error(self):

        self.gemini_mock.generate.side_effect = LLMUnavailableError("network error")
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.generate_jd(
                prompt="Some prompt",
                existing_draft=None,
            )
        self.assertIn("network", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_happy_path(self):

        self.gemini_mock.generate.return_value = '["Point 1", "Point 2"]'
        result = self.service.analyze_jd(
            jd_text="Job description text",
        )
        self.assertEqual(result, ["Point 1", "Point 2"])
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_non_json_raises_error(self):

        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.assertIn("malformed", str(cm.exception).lower())
        self.assertEqual(self.gemini_mock.generate.call_count, 2)  # original + repair retry

    def test_analyze_jd_non_list_raises_error(self):

        self.gemini_mock.generate.return_value = '{"key": "value"}'
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_non_string_item_raises_error(self):

        self.gemini_mock.generate.return_value = '["valid", 42]'
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_with_job_id_and_qdrant_success(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Summary of the job.",
            "company_name": "ORACLE",
            "about": "Nice",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"

        result = service.generate_jd(
            prompt="Create a JD",
            existing_draft=None,
            job_id=42,
        )

        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Senior Software Engineer", prompt)
        self.assertIn("Remote", prompt)
        self.assertIn("Python, AWS", prompt)
        self.assertIn("ORACLE", prompt)
        self.assertIn("Nice", prompt)

    def test_generate_jd_with_job_id_no_qdrant_raises(self):

        service = JDService(llm=self.gemini_mock, qdrant=None)
        with self.assertRaises(ValueError) as cm:
            service.generate_jd(
                prompt="Create a JD",
                existing_draft=None,
                job_id=42,
            )
        self.assertIn("job_id provided but Qdrant client is not configured", str(cm.exception))

    def test_generate_jd_with_job_id_not_found_raises(self):

        qdrant_mock = MagicMock()
        qdrant_mock.get.return_value = None
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        with self.assertRaises(ValueError) as cm:
            service.generate_jd(
                prompt="Create a JD",
                existing_draft=None,
                job_id=42,
            )
        self.assertIn("Job with ID 42 not found in Qdrant", str(cm.exception))

    def test_generate_jd_placeholder_replacement_company_name(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "Acme Corp",
            "about": "[About the Company]",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="We need a [Company Name] engineer",
            existing_draft=None,
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Acme Corp", prompt)
        self.assertNotIn("[Company Name]", prompt)

    def test_generate_jd_placeholder_replacement_about(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "[Company Name]",
            "about": "We are a tech company.",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="About: [About]",
            existing_draft="About the company: [About the Company]",
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("We are a tech company.", prompt)
        self.assertNotIn("[About]", prompt)
        self.assertNotIn("[About the Company]", prompt)

    def test_generate_jd_placeholder_replacement_both(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "Acme",
            "about": "Tech",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="Company: [Company Name], About: [About]",
            existing_draft=None,
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Acme", prompt)
        self.assertIn("Tech", prompt)
        self.assertNotIn("[Company Name]", prompt)
        self.assertNotIn("[About]", prompt)

    def test_generate_jd_company_and_about_resolved_from_qdrant(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
            "company_name": "ORACLE",
            "about": "Nice",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="Create a JD for [Company Name] about [About the Company]",
            existing_draft=None,
            job_id=2003,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 2003)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("ORACLE", prompt)
        self.assertIn("Nice", prompt)
        self.assertNotIn("[Company Name]", prompt)
        self.assertNotIn("[About the Company]", prompt)