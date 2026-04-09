"""Unit tests for utility functions."""

import json
import unittest
from unittest.mock import patch
from app.utils import parse_gemini_json


class TestParseGeminiJson(unittest.TestCase):
    """Test parse_gemini_json extraction of JSON from Gemini responses."""

    def test_plain_json_dict(self):
        """Valid JSON dict string should be parsed correctly."""
        generated = '{"feedback": "Great work", "score": 95}'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"feedback": "Great work", "score": 95})

    def test_plain_json_list(self):
        """Valid JSON list string should be parsed correctly."""
        generated = '[1, 2, 3, "four"]'
        result = parse_gemini_json(generated)
        self.assertEqual(result, [1, 2, 3, "four"])

    def test_markdown_fenced_json(self):
        """JSON inside markdown code fences should be stripped."""
        generated = '```json\n{"feedback": "Good", "score": 80}\n```'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"feedback": "Good", "score": 80})

    def test_string_value_with_braces(self):
        """JSON where a string value contains curly braces should not break depth counting."""
        generated = '{"feedback": "Use {curly} braces for templates", "score": 90}'
        result = parse_gemini_json(generated)
        self.assertEqual(
            result,
            {"feedback": "Use {curly} braces for templates", "score": 90},
        )

    def test_string_value_with_nested_braces(self):
        """String value containing JSON-like text should be treated as a plain string."""
        generated = r'{"note": "Ignore {\"nested\": \"value\"} inside string", "ok": true}'
        result = parse_gemini_json(generated)
        self.assertEqual(
            result,
            {"note": 'Ignore {"nested": "value"} inside string', "ok": True},
        )

    def test_escaped_quote_in_string(self):
        """JSON string containing escaped quotes should be parsed correctly."""
        generated = r'{"message": "He said \"hello\" to me"}'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"message": 'He said "hello" to me'})

    def test_json_embedded_in_prose(self):
        """JSON object embedded in prose text should be extracted."""
        generated = 'Here is the result: {"feedback": "Well done", "score": 85}. Thanks!'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"feedback": "Well done", "score": 85})

    def test_multiple_json_objects_in_text(self):
        """Two JSON objects in one string should return only the first valid object."""
        generated = 'First: {"a": 1}. Second: {"b": 2}.'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"a": 1})

    def test_nested_json_in_prose(self):
        """Nested JSON object embedded in prose should be extracted as whole outermost object."""
        generated = 'Some prose before {"outer": {"inner": 42}} and after.'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"outer": {"inner": 42}})

    def test_invalid_json_raises(self):
        """Completely non‑JSON text should raise json.JSONDecodeError."""
        generated = "This is just plain text with no JSON."
        with self.assertRaises(json.JSONDecodeError):
            parse_gemini_json(generated)

    def test_json_list_with_brace_strings(self):
        """JSON list where items contain curly braces should parse correctly."""
        generated = '[{"text": "item {1}"}, {"text": "item {2}"}]'
        result = parse_gemini_json(generated)
        self.assertEqual(result, [{"text": "item {1}"}, {"text": "item {2}"}])

    def test_prose_wrapped_array_of_objects(self):
        """JSON array of objects embedded in prose should be extracted."""
        generated = 'Here is the list: [{"a": 1}, {"b": 2}] and some trailing text.'
        result = parse_gemini_json(generated)
        self.assertEqual(result, [{"a": 1}, {"b": 2}])

    def test_mixed_object_and_array_prose(self):
        """When both object and array appear, the first occurring structure is selected."""
        # Array appears before object
        generated = 'Array first: [1, 2, 3]. Object later: {"x": 4}.'
        result = parse_gemini_json(generated)
        self.assertEqual(result, [1, 2, 3])
        # Object appears before array
        generated2 = 'Object first: {"x": 4}. Array later: [1, 2, 3].'
        result2 = parse_gemini_json(generated2)
        self.assertEqual(result2, {"x": 4})

    def test_nested_brace_text_inside_json_string(self):
        """JSON strings containing nested braces should be parsed as strings."""
        generated = r'{"message": "Ignore { nested } braces and also [ brackets ] inside string"}'
        result = parse_gemini_json(generated)
        self.assertEqual(result, {"message": "Ignore { nested } braces and also [ brackets ] inside string"})

    def test_array_with_nested_objects(self):
        """JSON array containing nested objects should be parsed correctly."""
        generated = '[{"id": 1, "data": {"value": "foo"}}, {"id": 2, "data": {"value": "bar"}}]'
        result = parse_gemini_json(generated)
        self.assertEqual(result, [{"id": 1, "data": {"value": "foo"}}, {"id": 2, "data": {"value": "bar"}}])

    def test_logging_does_not_include_raw_content(self):
        """Verify that raw generated content is not logged."""
        with patch('app.utils.logger') as mock_logger:
            # Successful parse
            generated = '{"feedback": "Great work", "score": 95}'
            result = parse_gemini_json(generated)
            self.assertEqual(result, {"feedback": "Great work", "score": 95})
            # Check debug log
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertNotIn('raw', call_args.kwargs)
            self.assertEqual(call_args.args[0], "Parsing Gemini response")
            self.assertIn('length', call_args.kwargs)
            self.assertIn('hash', call_args.kwargs)
            # Ensure hash is not the raw content
            self.assertNotEqual(call_args.kwargs['hash'], generated)
            # Reset mock
            mock_logger.reset_mock()
            # Failed parse
            generated = "not json"
            with self.assertRaises(json.JSONDecodeError):
                parse_gemini_json(generated)
            # Check error log
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            self.assertNotIn('raw', call_args.kwargs)
            self.assertEqual(call_args.args[0], "Failed to extract JSON from Gemini response")
            self.assertIn('length', call_args.kwargs)
            self.assertIn('hash', call_args.kwargs)
            # Ensure exception doc is redacted
            try:
                parse_gemini_json(generated)
            except json.JSONDecodeError as e:
                self.assertEqual(e.doc, "[REDACTED]")


if __name__ == "__main__":
    unittest.main()