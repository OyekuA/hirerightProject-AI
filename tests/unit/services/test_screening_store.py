

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.screening_store import BatchScreeningStore

class TestScreeningStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = BatchScreeningStore(self.temp_dir)

    def test_create_and_retrieve(self):

        record = self.store.create(
            total=3,
            job_ref={"job_id": 1, "job_version": 1},
            callback_url="https://example.com/callback",
        )
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.total, 3)
        self.assertEqual(record.job_ref, {"job_id": 1, "job_version": 1})
        self.assertEqual(record.callback_url, "https://example.com/callback")
        self.assertEqual(record.results, [])

        retrieved = self.store.get_by_batch_id(record.batch_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.batch_id, record.batch_id)
        self.assertEqual(retrieved.status, "pending")
        self.assertEqual(retrieved.total, 3)

    def test_update(self):

        record = self.store.create(total=2, job_ref={"jd_text": "some text"})
        self.store.update(record.batch_id, status="running")
        updated = self.store.get_by_batch_id(record.batch_id)
        self.assertEqual(updated.status, "running")

        self.store.update(record.batch_id, status="completed")
        completed = self.store.get_by_batch_id(record.batch_id)
        self.assertEqual(completed.status, "completed")

    def test_append_result(self):

        record = self.store.create(total=2, job_ref={"job_id": 1, "job_version": 1})

        self.store.append_result(record.batch_id, {
            "candidate_ref": "cand_1",
            "status": "scored",
            "fit_score": 85,
        })
        self.store.append_result(record.batch_id, {
            "candidate_ref": "cand_2",
            "status": "failed",
            "error": "CV parse error",
        })

        updated = self.store.get_by_batch_id(record.batch_id)
        self.assertEqual(len(updated.results), 2)
        self.assertEqual(updated.results[0]["candidate_ref"], "cand_1")
        self.assertEqual(updated.results[0]["fit_score"], 85)
        self.assertEqual(updated.results[1]["candidate_ref"], "cand_2")
        self.assertEqual(updated.results[1]["error"], "CV parse error")

    def test_get_missing_batch_id_returns_none(self):

        result = self.store.get_by_batch_id("non-existent-id")
        self.assertIsNone(result)

    def test_update_missing_batch_id_raises_key_error(self):

        with self.assertRaises(KeyError):
            self.store.update("non-existent-id", status="running")

    def test_append_result_missing_batch_id_raises_key_error(self):

        with self.assertRaises(KeyError):
            self.store.append_result("non-existent-id", {"candidate_ref": "x"})

    def test_logging_uses_structlog(self):

        with patch("app.services.screening_store._logger") as mock_logger:
            store = BatchScreeningStore(self.temp_dir)
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            self.assertEqual(call_args.args[0], "Batch screening store initialized")
            self.assertIn("store_path", call_args.kwargs)
            mock_logger.reset_mock()

            record = store.create(total=1, job_ref={"job_id": 1})
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertEqual(call_args.args[0], "Created screening batch record")
            self.assertIn("batch_id", call_args.kwargs)
            mock_logger.reset_mock()

            store.update(record.batch_id, status="running")
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertEqual(call_args.args[0], "Updated screening batch record")
            self.assertIn("batch_id", call_args.kwargs)
            mock_logger.reset_mock()

            store.append_result(record.batch_id, {"candidate_ref": "x"})
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            self.assertEqual(call_args.args[0], "Appended result to screening batch")
            self.assertIn("batch_id", call_args.kwargs)

if __name__ == "__main__":
    unittest.main()
