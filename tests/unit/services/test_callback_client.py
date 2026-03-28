"""Unit tests for CallbackClient."""

import json
import unittest
from unittest.mock import MagicMock, patch
import httpx
from app.services.callback_client import CallbackClient


class TestCallbackClient(unittest.TestCase):
    """Test the CallbackClient.send method."""

    def setUp(self):
        """Create a CallbackClient with test configuration."""
        self.client = CallbackClient(
            hmac_secret="test-secret",
            max_retries=3,
            retry_base_seconds=0,
        )
        self.sleep_patcher = patch("app.services.callback_client.time.sleep")
        self.sleep_patcher.start()
        self.validate_patcher = patch(
            "app.services.callback_client.validate_ingest_url"
        )
        self.mock_validate = self.validate_patcher.start()
        self.httpx_patcher = patch("app.services.callback_client.httpx.post")
        self.mock_post = self.httpx_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()
        self.validate_patcher.stop()
        self.httpx_patcher.stop()

    def test_successful_delivery_first_attempt(self):
        """When the first POST succeeds, send returns True and calls httpx.post once."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        self.mock_post.return_value = mock_response

        result = self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_123",
            entity_type="candidate",
            entity_id=42,
            status="processed",
            error=None,
        )
        self.assertTrue(result)
        self.mock_post.assert_called_once()
        call_kwargs = self.mock_post.call_args[1]
        self.assertIn("X-HireRight-Signature", call_kwargs["headers"])
        self.assertTrue(
            call_kwargs["headers"]["X-HireRight-Signature"].startswith("sha256=")
        )

    def test_retry_on_failure_then_success(self):
        """If first attempts fail but a later attempt succeeds, send returns True."""
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.raise_for_status.return_value = None
        self.mock_post.side_effect = [
            httpx.RequestError("connection failed", request=MagicMock()),
            httpx.RequestError("connection failed", request=MagicMock()),
            mock_success,
        ]

        result = self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_456",
            entity_type="job",
            entity_id=99,
            status="failed",
            error="Parsing error",
        )
        self.assertTrue(result)
        self.assertEqual(self.mock_post.call_count, 3)

    def test_all_retries_exhausted_returns_false(self):
        """If every attempt fails, send returns False."""
        self.mock_post.side_effect = httpx.RequestError("connection failed", request=MagicMock())

        result = self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_789",
            entity_type="candidate",
            entity_id=101,
            status="processed",
            error=None,
        )
        self.assertFalse(result)
        self.assertEqual(self.mock_post.call_count, 3)

    def test_ssrf_validation_failure_returns_false(self):
        """If validate_ingest_url raises ValueError, send returns False without HTTP call."""
        self.mock_validate.side_effect = ValueError("SSRF validation failed")
        result = self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_999",
            entity_type="job",
            entity_id=1,
            status="processed",
            error=None,
        )
        self.assertFalse(result)
        self.mock_post.assert_not_called()

    def test_hmac_signature_format(self):
        """The signature header should start with 'sha256=' and be hex."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        self.mock_post.return_value = mock_response

        self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_sig",
            entity_type="candidate",
            entity_id=5,
            status="processed",
            error=None,
        )
        call_kwargs = self.mock_post.call_args[1]
        signature_header = call_kwargs["headers"]["X-HireRight-Signature"]
        self.assertTrue(signature_header.startswith("sha256="))
        hex_part = signature_header[7:]
        self.assertTrue(all(c in "0123456789abcdef" for c in hex_part))
        self.assertEqual(len(hex_part), 64)

    def test_callback_body_and_headers(self):
        """Verify callback body does not contain timestamp, and required headers are present."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        self.mock_post.return_value = mock_response

        self.client.send(
            callback_url="https://example.com/callback",
            event_id="evt_body",
            entity_type="job",
            entity_id=99,
            status="processed",
            error=None,
        )

        call_kwargs = self.mock_post.call_args[1]
        body_bytes = call_kwargs["content"]
        body = json.loads(body_bytes.decode('utf-8'))
        self.assertNotIn("timestamp", body)
        self.assertEqual(body["event_id"], "evt_body")
        self.assertEqual(body["entity_type"], "job")
        self.assertEqual(body["entity_id"], 99)
        self.assertEqual(body["status"], "processed")
        self.assertIn("error", body)
        self.assertIsNone(body["error"])

        headers = call_kwargs["headers"]
        self.assertIn("X-HireRight-Timestamp", headers)
        self.assertIsInstance(headers["X-HireRight-Timestamp"], str)
        self.assertIn("X-HireRight-Signature", headers)
        self.assertTrue(headers["X-HireRight-Signature"].startswith("sha256="))