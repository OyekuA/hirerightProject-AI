"""Unit tests for Gemini client timeout and retry behavior."""

import unittest
from unittest.mock import MagicMock, patch, call
import time

from app.clients.gemini import GeminiClient, GeminiUnavailableError


class TestGeminiClientTimeout(unittest.TestCase):
    """Test that timeout and retry parameters are correctly used."""

    def test_generate_passes_timeout(self):
        """Generation call includes timeout parameter."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = MagicMock(text="response")
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "response")
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertNotIn('timeout', call_kwargs)
            client_call_kwargs = mock_client_cls.call_args.kwargs
            self.assertIn('http_options', client_call_kwargs)
            self.assertEqual(client_call_kwargs['http_options'].timeout, 45 * 1000)

    def test_embed_passes_timeout(self):
        """Embedding call includes timeout parameter."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_embedding = MagicMock(values=[0.1] * 768)
            mock_result = MagicMock(embeddings=[mock_embedding])
            mock_client.models.embed_content.return_value = mock_result
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=45,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.embed("text")
            self.assertEqual(len(result), 768)
            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args.kwargs
            self.assertNotIn('timeout', call_kwargs)
            client_call_kwargs = mock_client_cls.call_args.kwargs
            self.assertIn('http_options', client_call_kwargs)
            self.assertEqual(client_call_kwargs['http_options'].timeout, 45 * 1000)  # max of generation_timeout (45s) and embedding_timeout (30s)

    def test_generate_retry_on_exception(self):
        """Generation retries up to max_retries before failing."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls, \
             patch('time.sleep') as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = [
                Exception("First error"),
                Exception("Second error"),
                MagicMock(text="success"),
            ]
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.generate("prompt")
            self.assertEqual(result, "success")
            self.assertEqual(mock_client.models.generate_content.call_count, 3)
            mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    def test_generate_exhausts_retries_and_raises(self):
        """After max retries, GeminiUnavailableError is raised and breaker failure recorded."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls, \
             patch('time.sleep') as mock_sleep:
            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("Always fails")
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            with self.assertRaises(GeminiUnavailableError) as cm:
                client.generate("prompt")
            self.assertIn("Gemini generation failed after 3 attempts", str(cm.exception))
            self.assertEqual(mock_client.models.generate_content.call_count, 3)

    def test_embed_retry_on_exception(self):
        """Embedding retries up to max_retries before failing."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls, \
             patch('time.sleep') as mock_sleep:
            mock_client = MagicMock()
            mock_embedding = MagicMock(values=[0.1] * 768)
            mock_result = MagicMock(embeddings=[mock_embedding])
            mock_client.models.embed_content.side_effect = [
                Exception("First error"),
                Exception("Second error"),
                mock_result,
            ]
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
                generation_timeout=30,
                embedding_timeout=30,
                max_retries=2,
                retry_backoff_base=1.0,
            )
            result = client.embed("text")
            self.assertEqual(len(result), 768)
            self.assertEqual(mock_client.models.embed_content.call_count, 3)
            mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    def test_circuit_breaker_open_skips_retry(self):
        """If breaker is open, no retry attempts are made."""
        with patch('app.clients.gemini.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            client = GeminiClient(
                api_key="test",
                generation_cooldown=60,
                embedding_cooldown=60,
            )
            client._generation_breaker.record_failure()
            client._generation_breaker.record_failure()
            client._generation_breaker.record_failure()
            with self.assertRaises(GeminiUnavailableError) as cm:
                client.generate("prompt")
            self.assertIn("Generation circuit breaker is open", str(cm.exception))
            mock_client.models.generate_content.assert_not_called()

if __name__ == '__main__':
    unittest.main()