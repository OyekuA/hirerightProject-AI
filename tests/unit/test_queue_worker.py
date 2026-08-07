import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from app.main import process_queue_entry
from app.services.ingest_queue import IngestQueue, IngestQueueEntry
from app.services.ingestion_store import IngestionRecord


def _make_entry(entity_type: str = "candidate") -> IngestQueueEntry:
    return IngestQueueEntry(
        event_id="evt-test",
        entity_type=entity_type,
        entity_id=42,
        callback_url="https://example.com/callback",
        payload=(
            {"cv_url": "https://example.com/cv.pdf", "profile_data": {}}
            if entity_type == "candidate"
            else {"jd_text": "Job description", "metadata": {}}
        ),
        queue_retry_count=0,
        next_retry_at=datetime.now(timezone.utc).isoformat(),
        enqueued_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_settings(max_retries: int = 5, backoff_base: int = 60):
    settings = MagicMock()
    settings.INGEST_QUEUE_MAX_RETRIES = max_retries
    settings.INGEST_QUEUE_BACKOFF_BASE_SECONDS = backoff_base
    return settings


class TestProcessQueueEntry(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue = IngestQueue(
            queue_path=str(Path(self.temp_dir) / "queue"),
            dead_letter_path=str(Path(self.temp_dir) / "dead"),
        )
        self.store = MagicMock()
        self.callback_client = AsyncMock()
        self.callback_client.send.return_value = True

    def test_success_removes_entry_and_sends_success_callback(self):
        entry = _make_entry()
        entry_payload = entry.payload
        entry_dict = entry.to_dict()
        entry_dict["entity_type"] = entry.entity_type
        entry_dict["entity_id"] = entry.entity_id
        entry_dict["callback_url"] = entry.callback_url
        entry_dict["payload"] = entry_payload
        self.queue.enqueue(IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            attempt_count=4,
            callback_url=entry.callback_url,
            payload=entry_payload,
            created_at="",
            updated_at="",
        ), backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)

        success_record = IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="success",
            attempt_count=1,
            callback_url=entry.callback_url,
            payload=entry_payload,
            created_at="",
            updated_at="",
        )
        self.store.get_by_event_id.return_value = success_record

        async def _run_ingestion(**kwargs):
            return None

        with unittest.mock.patch("app.main.run_candidate_ingestion", new=_run_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertFalse(self.queue._path_for(entry.event_id).exists())
        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        self.callback_client.send.assert_awaited_once_with(
            callback_url=entry.callback_url,
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="success",
            error=None,
        )

    def test_failed_ingestion_requeues_entry(self):
        entry = _make_entry()
        record = IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            attempt_count=4,
            callback_url=entry.callback_url,
            payload=entry.payload,
            created_at="",
            updated_at="",
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)

        self.store.get_by_event_id.return_value = record

        async def _run_ingestion(**kwargs):
            return None

        with unittest.mock.patch("app.main.run_candidate_ingestion", new=_run_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertTrue(self.queue._path_for(entry.event_id).exists())
        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        self.callback_client.send.assert_not_awaited()

    def test_exception_in_ingestion_requeues_entry_not_stranded(self):
        entry = _make_entry()
        record = IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            attempt_count=4,
            callback_url=entry.callback_url,
            payload=entry.payload,
            created_at="",
            updated_at="",
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)

        async def _run_ingestion(**kwargs):
            raise KeyError("No ingestion record with event_id evt-test")

        with unittest.mock.patch("app.main.run_candidate_ingestion", new=_run_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertTrue(self.queue._path_for(entry.event_id).exists())
        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        self.callback_client.send.assert_not_awaited()

    def test_exception_after_max_retries_dead_letters_and_sends_failed_callback(self):
        entry = _make_entry()
        record = IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            attempt_count=4,
            callback_url=entry.callback_url,
            payload=entry.payload,
            created_at="",
            updated_at="",
        )
        self.queue.enqueue(record, backoff_base=0)
        entry.queue_retry_count = 5
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        due[0].queue_retry_count = 5

        async def _run_ingestion(**kwargs):
            raise KeyError("No ingestion record with event_id evt-test")

        with unittest.mock.patch("app.main.run_candidate_ingestion", new=_run_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertTrue(self.queue._dead_letter_path_for(entry.event_id).exists())
        self.assertFalse(self.queue._path_for(entry.event_id).exists())
        self.callback_client.send.assert_awaited_once_with(
            callback_url=entry.callback_url,
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            error="processing_error_max_queue_retries",
        )

    def test_failed_status_after_max_retries_dead_letters_with_callback(self):
        entry = _make_entry()
        record = IngestionRecord(
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            attempt_count=4,
            callback_url=entry.callback_url,
            payload=entry.payload,
            created_at="",
            updated_at="",
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        due[0].queue_retry_count = 5
        self.store.get_by_event_id.return_value = record

        async def _run_ingestion(**kwargs):
            return None

        with unittest.mock.patch("app.main.run_candidate_ingestion", new=_run_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertTrue(self.queue._dead_letter_path_for(entry.event_id).exists())
        self.assertFalse(self.queue._path_for(entry.event_id).exists())
        self.callback_client.send.assert_awaited_once_with(
            callback_url=entry.callback_url,
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            status="failed",
            error="max_queue_retries_exceeded",
        )

    def test_job_entity_type_dispatches_to_job_ingestion(self):
        entry = _make_entry(entity_type="job")
        record = IngestionRecord(
            event_id=entry.event_id,
            entity_type="job",
            entity_id=entry.entity_id,
            status="success",
            attempt_count=1,
            callback_url=entry.callback_url,
            payload=entry.payload,
            created_at="",
            updated_at="",
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        self.store.get_by_event_id.return_value = record

        called_with = {}

        async def _run_job_ingestion(**kwargs):
            called_with.update(kwargs)
            return None

        with unittest.mock.patch("app.main.run_job_ingestion", new=_run_job_ingestion):
            asyncio.run(process_queue_entry(
                due[0], self.queue, MagicMock(), MagicMock(), self.store,
                self.callback_client, _make_settings(),
            ))

        self.assertEqual(called_with["job_id"], 42)
        self.assertEqual(called_with["jd_text"], "Job description")
        self.assertIsNone(called_with["ingest_queue"])
        self.assertTrue(called_with["suppress_callback"])
        self.assertFalse(self.queue._path_for(entry.event_id).exists())


if __name__ == "__main__":
    unittest.main()
