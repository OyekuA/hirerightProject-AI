
import unittest
from app.utils.bias_masking import mask_candidate_for_scoring

class TestMaskCandidateForScoring(unittest.TestCase):

    def test_removes_name(self):

        payload = {"name": "Alice", "skills": ["Python"]}
        masked = mask_candidate_for_scoring(payload)
        self.assertNotIn("name", masked)

    def test_removes_single_token_location(self):

        payload = {"name": "Alice", "location": "Remote", "skills": ["Python"]}
        masked = mask_candidate_for_scoring(payload)
        self.assertNotIn("location", masked)

    def test_coarsens_location_with_comma(self):

        payload = {"name": "Alice", "location": "Berlin, Germany", "skills": ["Python"]}
        masked = mask_candidate_for_scoring(payload)
        self.assertEqual(masked["location"], "Germany")

    def test_coarsens_location_with_multiple_commas(self):

        payload = {"name": "Bob", "location": "Brooklyn, New York, USA"}
        masked = mask_candidate_for_scoring(payload)
        self.assertEqual(masked["location"], "USA")

    def test_leaves_other_fields_untouched(self):

        payload = {
            "name": "Charlie",
            "location": "London, UK",
            "skills": ["Python", "Docker"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Some text.",
            "employment_type": "full-time",
            "experience_level": "senior",
        }
        masked = mask_candidate_for_scoring(payload)
        self.assertEqual(masked["skills"], ["Python", "Docker"])
        self.assertEqual(masked["past_roles"], ["Engineer"])
        self.assertEqual(masked["raw_profile_summary"], "Some text.")
        self.assertEqual(masked["employment_type"], "full-time")
        self.assertEqual(masked["experience_level"], "senior")

    def test_does_not_mutate_input(self):

        original = {
            "name": "Alice",
            "location": "Berlin, Germany",
            "skills": ["Python"],
        }
        original_copy = dict(original)
        masked = mask_candidate_for_scoring(original)
        self.assertIsNot(masked, original)
        self.assertIn("name", original)
        self.assertEqual(original["name"], "Alice")
        self.assertEqual(original["location"], "Berlin, Germany")
        self.assertEqual(original, original_copy)

    def test_empty_payload(self):

        masked = mask_candidate_for_scoring({})
        self.assertEqual(masked, {})

    def test_no_name_key(self):

        payload = {"location": "New York, USA", "skills": ["Go"]}
        masked = mask_candidate_for_scoring(payload)
        self.assertEqual(masked["location"], "USA")
        self.assertEqual(masked["skills"], ["Go"])

    def test_location_not_a_string(self):

        payload = {"name": "X", "location": None, "skills": ["Rust"]}
        masked = mask_candidate_for_scoring(payload)
        self.assertNotIn("location", masked)
