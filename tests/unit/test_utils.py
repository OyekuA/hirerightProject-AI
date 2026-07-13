import json
import unittest
from unittest.mock import patch
from app.utils import parse_llm_json

class TestParseLLMJson(unittest.TestCase):

    def test_plain_json_dict(self):

        generated = '{"feedback": "Great work", "score": 95}'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"feedback": "Great work", "score": 95})

    def test_plain_json_list(self):

        generated = '[1, 2, 3, "four"]'
        result = parse_llm_json(generated)
        self.assertEqual(result, [1, 2, 3, "four"])

    def test_markdown_fenced_json(self):

        generated = '```json\n{"feedback": "Good", "score": 80}\n```'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"feedback": "Good", "score": 80})

    def test_string_value_with_braces(self):

        generated = '{"feedback": "Use {curly} braces for templates", "score": 90}'
        result = parse_llm_json(generated)
        self.assertEqual(
            result,
            {"feedback": "Use {curly} braces for templates", "score": 90},
        )

    def test_string_value_with_nested_braces(self):

        generated = r'{"note": "Ignore {\"nested\": \"value\"} inside string", "ok": true}'
        result = parse_llm_json(generated)
        self.assertEqual(
            result,
            {"note": 'Ignore {"nested": "value"} inside string', "ok": True},
        )

    def test_escaped_quote_in_string(self):

        generated = r'{"message": "He said \"hello\" to me"}'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"message": 'He said "hello" to me'})

    def test_json_embedded_in_prose(self):

        generated = 'Here is the result: {"feedback": "Well done", "score": 85}. Thanks!'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"feedback": "Well done", "score": 85})

    def test_multiple_json_objects_in_text(self):

        generated = 'First: {"a": 1}. Second: {"b": 2}.'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"a": 1})

    def test_nested_json_in_prose(self):

        generated = 'Some prose before {"outer": {"inner": 42}} and after.'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"outer": {"inner": 42}})

    def test_invalid_json_raises(self):

        generated = "This is just plain text with no JSON."
        with self.assertRaises(json.JSONDecodeError):
            parse_llm_json(generated)

    def test_json_list_with_brace_strings(self):

        generated = '[{"text": "item {1}"}, {"text": "item {2}"}]'
        result = parse_llm_json(generated)
        self.assertEqual(result, [{"text": "item {1}"}, {"text": "item {2}"}])

    def test_prose_wrapped_array_of_objects(self):

        generated = 'Here is the list: [{"a": 1}, {"b": 2}] and some trailing text.'
        result = parse_llm_json(generated)
        self.assertEqual(result, [{"a": 1}, {"b": 2}])

    def test_mixed_object_and_array_prose(self):

        generated = 'Array first: [1, 2, 3]. Object later: {"x": 4}.'
        result = parse_llm_json(generated)
        self.assertEqual(result, [1, 2, 3])
        generated2 = 'Object first: {"x": 4}. Array later: [1, 2, 3].'
        result2 = parse_llm_json(generated2)
        self.assertEqual(result2, {"x": 4})

    def test_nested_brace_text_inside_json_string(self):

        generated = r'{"message": "Ignore { nested } braces and also [ brackets ] inside string"}'
        result = parse_llm_json(generated)
        self.assertEqual(result, {"message": "Ignore { nested } braces and also [ brackets ] inside string"})

    def test_array_with_nested_objects(self):

        generated = '[{"id": 1, "data": {"value": "foo"}}, {"id": 2, "data": {"value": "bar"}}]'
        result = parse_llm_json(generated)
        self.assertEqual(result, [{"id": 1, "data": {"value": "foo"}}, {"id": 2, "data": {"value": "bar"}}])

    def test_logging_does_not_include_raw_content(self):

        with patch('app.utils.logger') as mock_logger:
            generated = '{"feedback": "Great work", "score": 95}'
            result = parse_llm_json(generated)
            self.assertEqual(result, {"feedback": "Great work", "score": 95})
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertNotIn('raw', call_args.kwargs)
            self.assertEqual(call_args.args[0], "Parsing LLM response")
            self.assertIn('length', call_args.kwargs)
            self.assertIn('hash', call_args.kwargs)
            self.assertNotEqual(call_args.kwargs['hash'], generated)
            mock_logger.reset_mock()
            generated = "not json"
            with self.assertRaises(json.JSONDecodeError):
                parse_llm_json(generated)
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            self.assertNotIn('raw', call_args.kwargs)
            self.assertEqual(call_args.args[0], "Failed to extract JSON from LLM response")
            self.assertIn('length', call_args.kwargs)
            self.assertIn('hash', call_args.kwargs)
            self.assertIn('starts_with', call_args.kwargs)
            self.assertIn('ends_with', call_args.kwargs)
            self.assertIn('looked_fenced', call_args.kwargs)
            try:
                parse_llm_json(generated)
            except json.JSONDecodeError as e:
                self.assertEqual(e.doc, "[REDACTED]")
                # pos should reflect actual length, not 0
                self.assertGreaterEqual(e.pos, 0)

    def test_invalid_json_error_contains_length_and_hash(self):

        generated = "This is just plain text with no JSON."
        with self.assertRaises(json.JSONDecodeError) as cm:
            parse_llm_json(generated)
        msg = str(cm.exception)
        self.assertIn("length=", msg)
        self.assertIn("hash=", msg)
        self.assertNotIn("char 0", msg)

if __name__ == "__main__":
    unittest.main()
