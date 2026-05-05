"""Unit tests for rate‑limit key extraction functions."""

import hashlib
import json
import unittest
from unittest.mock import MagicMock, patch

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


class TestCandidateIdKey(unittest.TestCase):
    """Test candidate_id_key synchronous function."""

    def test_candidate_id_present(self):
        """Key format: candidate:{fingerprint_with_entity_id}."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({"candidate_id": 123}).encode()
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"123").hexdigest()
        self.assertEqual(key, f"candidate:{expected_fp}")

    def test_candidate_id_missing(self):
        """Missing candidate_id yields unknown:{fingerprint}."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({}).encode()
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_candidate_id_non_dict_body(self):
        """Non‑dict JSON body falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps([1, 2, 3]).encode()
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_candidate_id_json_decode_error(self):
        """Malformed JSON raises exception, caught and returns unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = b"not valid json"
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_candidate_id_body_missing(self):
        """Missing _body attribute (AttributeError) falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        del request._body
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_candidate_id_body_empty(self):
        """Empty _body bytes falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = b""
        key = candidate_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


class TestJobIdKey(unittest.TestCase):
    """Test job_id_key synchronous function."""

    def test_job_id_present(self):
        """Key format: job:{fingerprint_with_entity_id}."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({"job_id": 456}).encode()
        key = job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"456").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")

    def test_job_id_missing(self):
        """Missing job_id yields unknown:{fingerprint}."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({}).encode()
        key = job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_job_id_without_api_key(self):
        """No API key falls back to IP fingerprint."""
        request = MagicMock(spec=Request)
        request.headers = {}
        request._body = json.dumps({"job_id": 789}).encode()
        with patch("app.routers._rate_limit_keys.get_remote_address") as mock_get:
            mock_get.return_value = "10.0.0.1"
            key = job_id_key(request)
        expected_fp = hashlib.sha256(b"10.0.0.1" + b"789").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")

    def test_job_id_body_missing(self):
        """Missing _body attribute falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        del request._body
        key = job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_job_id_body_empty(self):
        """Empty _body bytes falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = b""
        key = job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


class TestTargetIdKey(unittest.TestCase):
    """Test target_id_key synchronous function."""

    def test_target_id_present(self):
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({"target_id": "target_99"}).encode()
        key = target_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"target_99").hexdigest()
        self.assertEqual(key, f"target:{expected_fp}")

    def test_target_id_body_missing(self):
        """Missing _body attribute falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        del request._body
        key = target_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_target_id_body_empty(self):
        """Empty _body bytes falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = b""
        key = target_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


class TestCandidateOrJobIdKey(unittest.TestCase):
    """Test candidate_or_job_id_key synchronous function."""

    def test_candidate_context_present(self):
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps(
            {
                "candidate_context": {"candidate_id": 111},
                "job_context": {"job_id": 222},
            }
        ).encode()
        key = candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"111").hexdigest()
        self.assertEqual(key, f"candidate:{expected_fp}")

    def test_job_context_present(self):
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps(
            {
                "job_context": {"job_id": 333},
            }
        ).encode()
        key = candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret" + b"333").hexdigest()
        self.assertEqual(key, f"job:{expected_fp}")

    def test_neither_context_present(self):
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = json.dumps({}).encode()
        key = candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_body_missing(self):
        """Missing _body attribute falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        del request._body
        key = candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")

    def test_body_empty(self):
        """Empty _body bytes falls back to unknown."""
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "secret"}
        request._body = b""
        key = candidate_or_job_id_key(request)
        expected_fp = hashlib.sha256(b"secret").hexdigest()
        self.assertEqual(key, f"unknown:{expected_fp}")


if __name__ == "__main__":
    unittest.main()
