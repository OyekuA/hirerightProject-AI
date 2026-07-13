import unittest
from unittest.mock import MagicMock, patch, call
import time

from app.clients.llm import LLMClient, LLMUnavailableError

class TestLLMClientTimeout(unittest.TestCase):

    def test_generate_passes_timeout(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "response"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "response")
            mock_completion.assert_called_once_with(
                model="gemini/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": "prompt"}],
                timeout=45,
                temperature=0,
            )

    def test_embed_passes_timeout(self):

        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 768
        with patch('app.clients.llm.litellm.embedding', return_value=mock_response) as mock_embedding:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.embed("text")
            self.assertEqual(len(result), 768)
            mock_embedding.assert_called_once_with(
                model="gemini/text-embedding-004",
                input=["text"],
                dimensions=768,
                timeout=30,
            )

    def test_generate_passes_temperature_zero(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "response"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "response")
            self.assertEqual(mock_completion.call_args.kwargs["temperature"], 0)

    def test_generate_retry_on_exception(self):

        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            success_response = MagicMock()
            success_response.choices[0].message.content = "success"
            mock_completion.side_effect = [
                Exception("First error"),
                Exception("Second error"),
                success_response,
            ]
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "success")
            self.assertEqual(mock_completion.call_count, 3)
            mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    def test_generate_exhausts_retries_and_raises(self):

        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            mock_completion.side_effect = Exception("Always fails")
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            with self.assertRaises(LLMUnavailableError) as cm:
                client.generate("prompt")
            self.assertIn("LLM generation failed after 3 attempts", str(cm.exception))
            self.assertEqual(mock_completion.call_count, 3)

    def test_embed_retry_on_exception(self):

        with patch('app.clients.llm.litellm.embedding') as mock_embedding, \
             patch('time.sleep') as mock_sleep:
            success_response = MagicMock()
            success_response.data[0].embedding = [0.1] * 768
            mock_embedding.side_effect = [
                Exception("First error"),
                Exception("Second error"),
                success_response,
            ]
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.embed("text")
            self.assertEqual(len(result), 768)
            self.assertEqual(mock_embedding.call_count, 3)
            mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    def test_circuit_breaker_open_skips_retry(self):

        with patch('app.clients.llm.litellm.completion') as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
            )
            client._generation_breaker.record_failure()
            client._generation_breaker.record_failure()
            client._generation_breaker.record_failure()
            with self.assertRaises(LLMUnavailableError) as cm:
                client.generate("prompt")
            self.assertIn("Generation circuit breaker is open", str(cm.exception))
            mock_completion.assert_not_called()

    def test_generate_with_response_format(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"key": "value"}'
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion, \
             patch('app.clients.llm.litellm.get_supported_openai_params', return_value=["response_format"]):
            client = LLMClient(
                model="openai/gpt-4o-mini",
                embedding_model="openai/text-embedding-3-small",
                embedding_dimensions=1536,
                generation_cooldown=60,
                embedding_cooldown=60,
            )
            result = client.generate(
                "prompt",
                response_format={"type": "json_object"},
            )
            self.assertEqual(result, '{"key": "value"}')
            call_kwargs = mock_completion.call_args.kwargs
            self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
            self.assertTrue(call_kwargs["drop_params"])

    def test_generate_with_response_format_and_max_tokens_and_seed(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "output"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion, \
             patch('app.clients.llm.litellm.get_supported_openai_params', return_value=["response_format", "max_tokens", "seed"]):
            client = LLMClient(
                model="openai/gpt-4o-mini",
                embedding_model="openai/text-embedding-3-small",
                embedding_dimensions=1536,
                generation_cooldown=60,
                embedding_cooldown=60,
            )
            result = client.generate(
                "prompt",
                response_format={"type": "json_object"},
                max_tokens=4096,
                seed=42,
            )
            self.assertEqual(result, "output")
            call_kwargs = mock_completion.call_args.kwargs
            self.assertIn("response_format", call_kwargs)
            self.assertIn("max_tokens", call_kwargs)
            self.assertIn("seed", call_kwargs)
            self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
            self.assertEqual(call_kwargs["max_tokens"], 4096)
            self.assertEqual(call_kwargs["seed"], 42)
            self.assertTrue(call_kwargs["drop_params"])

    def test_generate_drops_unsupported_response_format(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "output"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion, \
             patch('app.clients.llm.litellm.get_supported_openai_params', return_value=[]):
            client = LLMClient(
                model="some-model",
                embedding_model="openai/text-embedding-3-small",
                embedding_dimensions=1536,
                generation_cooldown=60,
                embedding_cooldown=60,
            )
            result = client.generate(
                "prompt",
                response_format={"type": "json_object"},
            )
            self.assertEqual(result, "output")
            call_kwargs = mock_completion.call_args.kwargs
            self.assertNotIn("response_format", call_kwargs)
            self.assertNotIn("drop_params", call_kwargs)

    def test_generate_default_call_no_drop_params(self):

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "response"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "response")
            call_kwargs = mock_completion.call_args.kwargs
            self.assertNotIn("drop_params", call_kwargs)
            self.assertNotIn("response_format", call_kwargs)
            self.assertNotIn("max_tokens", call_kwargs)
            self.assertNotIn("seed", call_kwargs)

    def test_generate_honors_retry_after_on_429(self):

        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            rate_limit_exc = Exception("Rate limit")
            rate_limit_exc.status_code = 429
            rate_limit_exc.response_headers = {"Retry-After": "5"}
            success_response = MagicMock()
            success_response.choices[0].message.content = "success"
            mock_completion.side_effect = [rate_limit_exc, success_response]

            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=1,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "success")
            mock_sleep.assert_called_once_with(5)

    def test_generate_no_retry_after_falls_back_to_exponential(self):

        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            mock_completion.side_effect = [Exception("Generic error"), MagicMock()]
            success_response = MagicMock()
            success_response.choices[0].message.content = "success"
            mock_completion.side_effect = [Exception("Generic error"), success_response]

            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
                embedding_dimensions=768,
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=1,
                retry_backoff_base=2.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "success")
            mock_sleep.assert_called_once_with(2.0)

if __name__ == '__main__':
    unittest.main()
