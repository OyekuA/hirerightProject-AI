"""Unit tests for validate_ingest_url."""

import socket
import unittest
from unittest.mock import patch, MagicMock
import httpx
from app.utils.ingestion import validate_ingest_url, fetch_and_parse_cv


class TestValidateIngestUrl(unittest.TestCase):
    """Test the SSRF validation logic."""

    def test_http_url_raises_value_error(self):
        """Only HTTPS URLs are allowed."""
        with self.assertRaises(ValueError) as cm:
            validate_ingest_url("http://example.com/file.pdf")
        self.assertIn("HTTPS", str(cm.exception))

    def test_private_ip_raises_value_error(self):
        """URLs resolving to private IP addresses should be rejected."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("192.168.1.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://internal.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_loopback_ip_raises_value_error(self):
        """URLs resolving to loopback addresses should be rejected."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("127.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://localhost/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))

    def test_reserved_ip_raises_value_error(self):
        """URLs resolving to reserved IP addresses should be rejected."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("240.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://reserved.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_multicast_ip_raises_value_error(self):
        """URLs resolving to multicast IP addresses should be rejected."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("224.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://multicast.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_unspecified_ip_raises_value_error(self):
        """URLs resolving to unspecified IP addresses should be rejected."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("0.0.0.0", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://unspecified.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_valid_public_ip_no_exception(self):
        """A URL that resolves to a public IP should pass validation."""
        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("8.8.8.8", 0)),
            ]
            try:
                validate_ingest_url("https://example.com/file.pdf")
            except ValueError as e:
                self.fail(f"validate_ingest_url raised unexpected ValueError: {e}")
            mock_getaddrinfo.assert_called_once()


class TestFetchAndParseCv(unittest.TestCase):
    """Test fetch_and_parse_cv redirect handling."""

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_redirect_to_private_ip_raises_value_error(self, mock_get_settings, mock_client_class, mock_validate):
        """Open redirect to a private IP should be caught by validation."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        # Mock validation to raise ValueError for private IP
        def validate_side_effect(url):
            if "192.168.1.1" in url:
                raise ValueError("Resolved IP 192.168.1.1 is in a prohibited network range")
        mock_validate.side_effect = validate_side_effect

        # Mock HTTP client behavior
        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        # First response is a redirect to private IP
        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://192.168.1.1/file.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()
        # Simulate stream context manager
        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = redirect_response
        mock_stream.__exit__.return_value = None

        with self.assertRaises(ValueError) as cm:
            fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertIn("prohibited network", str(cm.exception))
        # Ensure validation was called for the redirect target
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://192.168.1.1/file.pdf", call_urls)

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_too_many_redirects_raises_value_error(self, mock_get_settings, mock_client_class, mock_validate):
        """Exceeding MAX_REDIRECTS (5) should raise ValueError."""
        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        # Mock validation to pass
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        # Create a chain of redirect responses
        responses = []
        for i in range(6):  # 6 redirects > MAX_REDIRECTS (5)
            resp = MagicMock()
            resp.status_code = 302
            resp.headers = {"Location": f"https://example{i}.com/file.pdf"}
            resp.url = MagicMock()
            resp.url.host = f"example{i}.com"
            resp.url.__str__.return_value = f"https://example{i}.com/file.pdf"
            resp.raise_for_status = MagicMock()
            responses.append(resp)
        # Final response (non-redirect) will never be reached
        # Mock stream to return each redirect sequentially
        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.side_effect = responses
        mock_stream.__exit__.return_value = None

        with self.assertRaises(ValueError) as cm:
            fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertIn("Too many redirects", str(cm.exception))
    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.fitz")
    def test_absolute_redirect_success(self, mock_fitz, mock_get_settings, mock_client_class, mock_validate):
        """Absolute redirect leads to successful PDF download and parsing."""
        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        # Redirect response
        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://example2.com/file.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()

        # Final PDF response
        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.pdf"
        final_response.url.host = "example2.com"
        final_response.url.__str__.return_value = "https://example2.com/file.pdf"
        final_response.raise_for_status = MagicMock()
        # Simulate PDF content
        final_response.iter_bytes.return_value = [b"%PDF-1.4 dummy content"]

        # Mock stream to return redirect then final
        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.side_effect = [redirect_response, final_response]
        mock_stream.__exit__.return_value = None

        # Mock fitz parsing
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "parsed text"
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        result = fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertEqual(result, "parsed text")
        # Ensure validation called for initial URL, redirect target URL, and final resolved URL
        self.assertGreaterEqual(mock_validate.call_count, 3)
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://example.com/file.pdf", call_urls)
        self.assertIn("https://example2.com/file.pdf", call_urls)
        # Ensure final resolved URL validation (same as redirect target) occurred after the redirect
        self.assertEqual(call_urls[-1], "https://example2.com/file.pdf")

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.fitz")
    def test_relative_redirect_success(self, mock_fitz, mock_get_settings, mock_client_class, mock_validate):
        """Relative redirect leads to successful PDF download and parsing."""
        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        # Redirect response with relative Location
        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "/new.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()

        # Final PDF response
        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.url = MagicMock()
        final_response.url.path = "/new.pdf"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/new.pdf"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"%PDF-1.4 dummy content"]

        # Mock stream to return redirect then final
        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.side_effect = [redirect_response, final_response]
        mock_stream.__exit__.return_value = None

        # Mock fitz parsing
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "parsed text"
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        result = fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertEqual(result, "parsed text")
        # Ensure validation called for initial URL, redirect target URL, and final resolved URL
        self.assertGreaterEqual(mock_validate.call_count, 3)
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://example.com/file.pdf", call_urls)
        self.assertIn("https://example.com/new.pdf", call_urls)
        # Ensure final resolved URL validation (same as redirect target) occurred after the redirect
        self.assertEqual(call_urls[-1], "https://example.com/new.pdf")
