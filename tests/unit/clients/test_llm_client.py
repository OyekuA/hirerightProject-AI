"""Unit tests for LLM client timeout and retry behavior."""

import unittest
from unittest.mock import MagicMock, patch, call
import time

from app.clients.llm import LLMClient, LLMUnavailableError


class TestLLMClientTimeout(unittest.TestCase):
    """Test that timeout and retry parameters are correctly used."""

    def test_generate_passes_timeout(self):
        """Generation call includes timeout parameter."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "response"
        with patch('app.clients.llm.litellm.completion', return_value=mock_response) as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
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
            )

    def test_embed_passes_timeout(self):
        """Embedding call includes timeout parameter."""
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1] * 768
        with patch('app.clients.llm.litellm.embedding', return_value=mock_response) as mock_embedding:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
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
                timeout=30,
            )

    def test_generate_retry_on_exception(self):
        """Generation retries up to max_retries before failing."""
        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            mock_completion.side_effect = [
                Exception("First error"),
                Exception("Second error"),
                MagicMock(text="success"),  # will be overridden below
            ]
            # Set up the successful response properly
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
        """After max retries, LLMUnavailableError is raised and breaker failure recorded."""
        with patch('app.clients.llm.litellm.completion') as mock_completion, \
             patch('time.sleep') as mock_sleep:
            mock_completion.side_effect = Exception("Always fails")
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
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
        """Embedding retries up to max_retries before failing."""
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
        """If breaker is open, no retry attempts are made."""
        with patch('app.clients.llm.litellm.completion') as mock_completion:
            client = LLMClient(
                model="gemini/gemini-2.5-flash-lite",
                embedding_model="gemini/text-embedding-004",
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

if __name__ == '__main__':
    unittest.main()
