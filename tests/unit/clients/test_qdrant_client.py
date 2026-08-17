import unittest
from unittest.mock import MagicMock, patch

from qdrant_client.http.exceptions import ResponseHandlingException

from app.clients.qdrant import QdrantClient, QdrantUnavailableError, MISSING


class TestQdrantTransportGuarding(unittest.TestCase):

    def _client_with_failing_sdk(self, exc):
        client = QdrantClient(host="127.0.0.1", port=6333)
        sdk = MagicMock()
        sdk.retrieve.side_effect = exc
        sdk.query_points.side_effect = exc
        client._client = sdk
        return client

    def test_get_wraps_response_handling_exception(self):
        client = self._client_with_failing_sdk(ResponseHandlingException("boom"))
        with self.assertRaises(QdrantUnavailableError):
            client.get("candidates", 1)

    def test_get_wraps_connection_error(self):
        client = self._client_with_failing_sdk(ConnectionError("refused"))
        with self.assertRaises(QdrantUnavailableError):
            client.get("candidates", 1)

    def test_get_with_vector_wraps_timeout(self):
        from httpx import ReadTimeout
        client = self._client_with_failing_sdk(ReadTimeout("timed out"))
        with self.assertRaises(QdrantUnavailableError):
            client.get_with_vector("candidates", 1)

    def test_search_wraps_transport_error(self):
        client = self._client_with_failing_sdk(ResponseHandlingException("boom"))
        with self.assertRaises(QdrantUnavailableError):
            client.search("candidates", [0.0] * 4, limit=5)

    def test_get_returns_missing_when_absent(self):
        client = QdrantClient(host="127.0.0.1", port=6333)
        sdk = MagicMock()
        sdk.retrieve.return_value = []
        client._client = sdk
        self.assertIs(client.get("candidates", 1), MISSING)

    def test_successful_call_not_wrapped(self):
        client = QdrantClient(host="127.0.0.1", port=6333)
        sdk = MagicMock()
        sdk.retrieve.return_value = []
        client._client = sdk
        result = client.get("candidates", 1)
        self.assertIs(result, MISSING)
