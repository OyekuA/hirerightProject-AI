"""Unit tests for rate‑limit key extraction functions."""

import hashlib
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request

from app.routers._rate_limit_keys import (
    _fingerprint,
    candidate_id_key,
    job_id_key,
    target_id_key,
    candidate_or_job_id_key,
)


class TestFingerprint(unittest.TestCase):
    """Test the internal _fingerprint helper."""

    def test_fingerprint_with_api_key(self):
        """SHA‑256 fingerprint of API key."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret123"}
        with patch("app.routers._rate_limit_keys.get_remote_address") as mock_get:
            mock_get.return_value = "192.168.1.1"
            fp = _fingerprint(request)
        expected = hashlib.sha256(b"secret123").hexdigest()
        self.assertEqual(fp, expected)

    def test_fingerprint_with_ip_fallback(self):
        """Fall back to client IP when no API key."""
        request = MagicMock(spec=Request)
        request.headers = {}
        with patch("app.routers._rate_limit_keys.get_remote_address") as mock_get:
            mock_get.return_value = "192.168.1.1"
            fp = _fingerprint(request)
        expected = hashlib.sha256(b"192.168.1.1").hexdigest()
        self.assertEqual(fp, expected)

    def test_fingerprint_with_entity_id(self):
        """Entity ID is concatenated to the base bytes."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret123"}
        with patch("app.routers._rate_limit_keys.get_remote_address"):
            fp = _fingerprint(request, entity_id="candidate_42")
        expected = hashlib.sha256(b"secret123" + b"candidate_42").hexdigest()
        self.assertEqual(fp, expected)

    def test_fingerprint_different_keys_give_different_hashes(self):
        """Different API keys produce distinct fingerprints."""
        request1 = MagicMock(spec=Request)
        request1.headers = {"X-API-Key": "key1"}
        request2 = MagicMock(spec=Request)
        request2.headers = {"X-API-Key": "key2"}
        with patch("app.routers._rate_limit_keys.get_remote_address"):
            fp1 = _fingerprint(request1)
            fp2 = _fingerprint(request2)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_same_key_same_hash(self):
        """Same API key yields identical fingerprint across requests."""
        request1 = MagicMock(spec=Request)
        request1.headers = {"X-API-Key": "same"}
        request2 = MagicMock(spec=Request)
        request2.headers = {"X-API-Key": "same"}
        with patch("app.routers._rate_limit_keys.get_remote_address"):
            fp1 = _fingerprint(request1)
            fp2 = _fingerprint(request2)
        self.assertEqual(fp1, fp2)


class TestCandidateIdKey(unittest.IsolatedAsyncioTestCase):
    """Test candidate_id_key async function."""

    async def test_candidate_id_present(self):
        """Key format: candidate:{fingerprint_with_entity_id}."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={"candidate_id": 123})
        key = await candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"123").hexdigest()
        self.assertEqual(key, f"candidate:{expected_fp}")

    async def test_candidate_id_missing(self):
        """Missing candidate_id yields unknown:{fingerprint}."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={})
        key = await candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    async def test_candidate_id_non_dict_body(self):
        """Non‑dict JSON body falls back to unknown."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value=[1, 2, 3])
        key = await candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    async def test_json_decode_error(self):
        """Malformed JSON raises exception, caught and returns unknown."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(side_effect=json.JSONDecodeError("Expecting value", "", 0))
        key = await candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


class TestJobIdKey(unittest.IsolatedAsyncioTestCase):
    """Test job_id_key async function."""

    async def test_job_id_present(self):
        """Key format: job:{fingerprint_with_entity_id}."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={"job_id": 456})
        key = await job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"456").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")

    async def test_job_id_missing(self):
        """Missing job_id yields unknown:{fingerprint}."""
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={})
        key = await job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    async def test_job_id_without_api_key(self):
        """No API key falls back to IP fingerprint."""
        request = AsyncMock(spec=Request)
        request.headers = {}
        request.json = AsyncMock(return_value={"job_id": 789})
        with patch("app.routers._rate_limit_keys.get_remote_address") as mock_get:
            mock_get.return_value = "10.0.0.1"
            key = await job_id_key(request)
        expected_fp = hashlib.sha256(b"10.0.0.1" + b"789").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")


class TestTargetIdKey(unittest.IsolatedAsyncioTestCase):
    """Test target_id_key async function."""

    async def test_target_id_present(self):
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={"target_id": "target_99"})
        key = await target_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"target_99").hexdigest()
        self.assertEqual(key, f"target:{expected_fp}")


class TestCandidateOrJobIdKey(unittest.IsolatedAsyncioTestCase):
    """Test candidate_or_job_id_key async function."""

    async def test_candidate_context_present(self):
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(
            return_value={
                "candidate_context": {"candidate_id": 111},
                "job_context": {"job_id": 222},
            }
        )
        key = await candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"111").hexdigest()
        self.assertEqual(key, f"candidate:{expected_fp}")

    async def test_job_context_present(self):
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(
            return_value={
                "job_context": {"job_id": 333},
            }
        )
        key = await candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"333").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")

    async def test_neither_context_present(self):
        request = AsyncMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request.json = AsyncMock(return_value={})
        key = await candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


if __name__ == "__main__":
    unittest.main()