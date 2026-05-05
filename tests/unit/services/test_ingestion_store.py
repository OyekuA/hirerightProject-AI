"""Unit tests for ingestion status store logging safety."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.ingestion_store import IngestionStatusStore


class TestIngestionStoreLogging(unittest.TestCase):
    """Verify that ingestion store logging never raises runtime exceptions."""

    def setUp(self):
        # Create a temporary directory for store files
        self.temp_dir = tempfile.mkdtemp()
        self.store = IngestionStatusStore(self.temp_dir)

    def test_create_logging_no_exception(self):
        """Creating a record should log without raising TypeError."""
        # If logging raises an exception, the test will fail.
        record = self.store.create(
            "candidate", 123, "https://example.com/callback",
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
        )
        self.assertEqual(record.entity_type, "candidate")
        self.assertEqual(record.entity_id, 123)

    def test_update_logging_no_exception(self):
        """Updating a record should log without raising TypeError."""
        record = self.store.create(
            "candidate", 123, "https://example.com/callback",
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
        )
        # If logging raises an exception, the test will fail.
        self.store.update(record.event_id, status="running")
        updated = self.store.get_by_event_id(record.event_id)
        self.assertEqual(updated.status, "running")

    def test_corrupt_file_scan_logging_no_exception(self):
        """Scanning a directory with corrupt JSON files should log warnings without raising TypeError."""
        # Write a corrupt JSON file directly into the store directory
        corrupt_path = Path(self.temp_dir) / "corrupt.json"
        corrupt_path.write_text("{ invalid json", encoding="utf-8")
        # Calling get_by_entity or get_all_incomplete will encounter the corrupt file
        # and log a warning. Ensure no exception is raised.
        result = self.store.get_by_entity("candidate", 999)
        self.assertIsNone(result)
        incomplete = self.store.get_all_incomplete()
        self.assertEqual(len(incomplete), 0)

    def test_logging_uses_structlog(self):
        """Verify that logging calls are made with keyword arguments (structlog compatible)."""
        with patch('app.services.ingestion_store._logger') as mock_logger:
            store = IngestionStatusStore(self.temp_dir)
            # Check that the init log includes store_path keyword
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            self.assertEqual(call_args.args[0], "Ingestion status store initialized")
            self.assertIn('store_path', call_args.kwargs)
            mock_logger.reset_mock()
            # Create a record and check debug log
            record = store.create(
                "job", 456, "https://example.com/callback",
                payload={"jd_text": "Some JD", "metadata": {}},
            )
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertEqual(call_args.args[0], "Created ingestion record")
            self.assertIn('event_id', call_args.kwargs)
            self.assertIn('entity_type', call_args.kwargs)
            self.assertIn('entity_id', call_args.kwargs)
            mock_logger.reset_mock()
            # Update record
            store.update(record.event_id, status="success")
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertEqual(call_args.args[0], "Updated ingestion record")
            self.assertIn('event_id', call_args.kwargs)
            # Corrupt file warning
            corrupt_path = Path(self.temp_dir) / "corrupt2.json"
            corrupt_path.write_text("{ invalid", encoding="utf-8")
            store.get_by_entity("job", 999)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            self.assertEqual(call_args.args[0], "Skipping corrupt status file")
            self.assertIn('file_path', call_args.kwargs)
            self.assertIn('error', call_args.kwargs)


    def test_payload_is_persisted_and_retrievable(self):
        """Payload passed to create() should be stored and retrievable via get_by_event_id."""
        payload = {"cv_url": "https://example.com/cv.pdf", "profile_data": {"name": "Alice"}}
        record = self.store.create("candidate", 1, "https://example.com/cb", payload=payload)

        self.assertEqual(record.payload, payload)

        retrieved = self.store.get_by_event_id(record.event_id)
        self.assertEqual(retrieved.payload, payload)

    def test_payload_defaults_to_none(self):
        """Creating a record without payload should store None and survive round-trip."""
        record = self.store.create("job", 2, "https://example.com/cb")
        self.assertIsNone(record.payload)

        retrieved = self.store.get_by_event_id(record.event_id)
        self.assertIsNone(retrieved.payload)


if __name__ == "__main__":
    unittest.main()