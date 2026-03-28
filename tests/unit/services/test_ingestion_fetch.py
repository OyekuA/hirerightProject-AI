"""Unit tests for validate_ingest_url."""

import socket
import unittest
from unittest.mock import patch
from app.services.ingestion_fetch import validate_ingest_url


class TestValidateIngestUrl(unittest.TestCase):
    """Test the SSRF validation logic."""

    def test_http_url_raises_value_error(self):
        """Only HTTPS URLs are allowed."""
        with self.assertRaises(ValueError) as cm:
            validate_ingest_url("http://example.com/file.pdf")
        self.assertIn("HTTPS", str(cm.exception))

    def test_private_ip_raises_value_error(self):
        """URLs resolving to private IP addresses should be rejected."""
        with patch("app.services.ingestion_fetch.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("192.168.1.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://internal.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_loopback_ip_raises_value_error(self):
        """URLs resolving to loopback addresses should be rejected."""
        with patch("app.services.ingestion_fetch.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("127.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://localhost/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))

    def test_valid_public_ip_no_exception(self):
        """A URL that resolves to a public IP should pass validation."""
        with patch("app.services.ingestion_fetch.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("8.8.8.8", 0)),
            ]
            try:
                validate_ingest_url("https://example.com/file.pdf")
            except ValueError as e:
                self.fail(f"validate_ingest_url raised unexpected ValueError: {e}")
            mock_getaddrinfo.assert_called_once()