

import io
import socket
import unittest
from unittest.mock import patch, MagicMock
import httpx
from app.utils.ingestion import validate_ingest_url, fetch_and_parse_cv

class TestValidateIngestUrl(unittest.TestCase):

    def test_http_url_raises_value_error(self):

        with self.assertRaises(ValueError) as cm:
            validate_ingest_url("http://example.com/file.pdf")
        self.assertIn("HTTPS", str(cm.exception))

    def test_private_ip_raises_value_error(self):

        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("192.168.1.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://internal.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_loopback_ip_raises_value_error(self):

        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("127.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://localhost/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))

    def test_reserved_ip_raises_value_error(self):

        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("240.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://reserved.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_multicast_ip_raises_value_error(self):

        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("224.0.0.1", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://multicast.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_unspecified_ip_raises_value_error(self):

        with patch("app.utils.ingestion.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, None, None, None, ("0.0.0.0", 0)),
            ]
            with self.assertRaises(ValueError) as cm:
                validate_ingest_url("https://unspecified.example.com/file.pdf")
            self.assertIn("prohibited network", str(cm.exception))
            mock_getaddrinfo.assert_called_once()

    def test_valid_public_ip_no_exception(self):

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

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_redirect_to_private_ip_raises_value_error(self, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        def validate_side_effect(url):
            if "192.168.1.1" in url:
                raise ValueError("Resolved IP 192.168.1.1 is in a prohibited network range")
        mock_validate.side_effect = validate_side_effect

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://192.168.1.1/file.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()
        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = redirect_response
        mock_stream.__exit__.return_value = None

        with self.assertRaises(ValueError) as cm:
            fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertIn("prohibited network", str(cm.exception))
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://192.168.1.1/file.pdf", call_urls)

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_too_many_redirects_raises_value_error(self, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

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

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "https://example2.com/file.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.pdf"
        final_response.url.host = "example2.com"
        final_response.url.__str__.return_value = "https://example2.com/file.pdf"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"%PDF-1.4 dummy content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.side_effect = [redirect_response, final_response]
        mock_stream.__exit__.return_value = None

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "parsed text"
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        result = fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertEqual(result, "parsed text")
        self.assertGreaterEqual(mock_validate.call_count, 3)
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://example.com/file.pdf", call_urls)
        self.assertIn("https://example2.com/file.pdf", call_urls)
        self.assertEqual(call_urls[-1], "https://example2.com/file.pdf")

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.fitz")
    def test_relative_redirect_success(self, mock_fitz, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "/new.pdf"}
        redirect_response.url = MagicMock()
        redirect_response.url.host = "example.com"
        redirect_response.url.__str__.return_value = "https://example.com/file.pdf"
        redirect_response.raise_for_status = MagicMock()

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.url = MagicMock()
        final_response.url.path = "/new.pdf"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/new.pdf"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"%PDF-1.4 dummy content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.side_effect = [redirect_response, final_response]
        mock_stream.__exit__.return_value = None

        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "parsed text"
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        result = fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertEqual(result, "parsed text")
        self.assertGreaterEqual(mock_validate.call_count, 3)
        call_urls = [call[0][0] for call in mock_validate.call_args_list]
        self.assertIn("https://example.com/file.pdf", call_urls)
        self.assertIn("https://example.com/new.pdf", call_urls)
        self.assertEqual(call_urls[-1], "https://example.com/new.pdf")

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.docx.Document")
    def test_docx_parsing_success(self, mock_docx_document, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.docx"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.docx"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"PK\x03\x04 dummy docx content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_docx_document.return_value = mock_doc

        result = fetch_and_parse_cv("https://example.com/file.docx")
        self.assertEqual(result, "First paragraph\nSecond paragraph")
        mock_docx_document.assert_called_once()
        args, _ = mock_docx_document.call_args
        self.assertIsInstance(args[0], io.BytesIO)

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.fitz")
    def test_pdf_empty_content_rejection(self, mock_fitz, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.pdf"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.pdf"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"%PDF-1.4 dummy content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc

        with self.assertRaises(RuntimeError) as cm:
            fetch_and_parse_cv("https://example.com/file.pdf")
        self.assertIn("image-based or scanned", str(cm.exception))
        self.assertIn("text-based PDF", str(cm.exception))

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_txt_parsing_success(self, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "text/plain"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.txt"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.txt"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"Hello, world!"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        result = fetch_and_parse_cv("https://example.com/file.txt")
        self.assertEqual(result, "Hello, world!")

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_unsupported_extension_rejection(self, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/octet-stream"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.exe"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.exe"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"binary content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        with self.assertRaises(ValueError) as cm:
            fetch_and_parse_cv("https://example.com/file.exe")
        self.assertIn("supported extension", str(cm.exception))

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    @patch("app.utils.ingestion.docx.Document")
    def test_docx_empty_content_rejection(self, mock_docx_document, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.docx"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.docx"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"PK\x03\x04 dummy docx content"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        mock_para = MagicMock()
        mock_para.text = "   "
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_docx_document.return_value = mock_doc

        with self.assertRaises(RuntimeError) as cm:
            fetch_and_parse_cv("https://example.com/file.docx")
        self.assertIn("no extractable text", str(cm.exception).lower())

    @patch("app.utils.ingestion.validate_ingest_url")
    @patch("app.utils.ingestion.httpx.Client")
    @patch("app.config.get_settings")
    def test_txt_empty_content_rejection(self, mock_get_settings, mock_client_class, mock_validate):

        mock_settings = MagicMock()
        mock_settings.MAX_INGEST_FILE_MB = 10
        mock_settings.INGEST_FETCH_TIMEOUT_SECONDS = 30
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = None

        mock_client = mock_client_class.return_value
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = None

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "text/plain"}
        final_response.url = MagicMock()
        final_response.url.path = "/file.txt"
        final_response.url.host = "example.com"
        final_response.url.__str__.return_value = "https://example.com/file.txt"
        final_response.raise_for_status = MagicMock()
        final_response.iter_bytes.return_value = [b"   \n  \t  \n"]

        mock_stream = mock_client.stream.return_value
        mock_stream.__enter__.return_value = final_response
        mock_stream.__exit__.return_value = None

        with self.assertRaises(RuntimeError) as cm:
            fetch_and_parse_cv("https://example.com/file.txt")
        self.assertIn("no extractable text", str(cm.exception).lower())
