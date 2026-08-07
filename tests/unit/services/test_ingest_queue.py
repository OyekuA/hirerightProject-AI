

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.ingest_queue import IngestQueue, IngestQueueEntry
from app.services.ingestion_store import IngestionRecord

import uuid

_sentinel = object()

def _make_record(
    entity_type: str = "candidate",
    entity_id: int = 1,
    payload: object = _sentinel,
) -> IngestionRecord:

    final_payload = payload if payload is not _sentinel else {"cv_url": "https://example.com/cv.pdf", "profile_data": {}}
    return IngestionRecord(
        event_id=str(uuid.uuid4()),
        entity_type=entity_type,  # type: ignore
        entity_id=entity_id,
        status="failed",
        attempt_count=4,
        callback_url="https://example.com/callback",
        error_summary="SomeError: something went wrong",
        callback_delivery_failed=False,
        payload=final_payload,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

class TestIngestQueue(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    def _assert_file_exists(self, path: Path, msg: str = ""):
        self.assertTrue(path.exists(), msg or f"Expected file to exist: {path}")

    def _assert_file_not_exists(self, path: Path, msg: str = ""):
        self.assertFalse(path.exists(), msg or f"Expected file to be absent: {path}")

    def _read_entry(self, event_id: str) -> dict:
        path = self.queue._path_for(event_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_enqueue_creates_file_with_correct_fields(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)

        path = self.queue._path_for(record.event_id)
        self._assert_file_exists(path)

        data = self._read_entry(record.event_id)
        self.assertEqual(data["event_id"], record.event_id)
        self.assertEqual(data["entity_type"], "candidate")
        self.assertEqual(data["entity_id"], 1)
        self.assertEqual(data["queue_retry_count"], 0)
        self.assertIn("next_retry_at", data)
        next_retry = datetime.fromisoformat(data["next_retry_at"])
        self.assertGreater(next_retry, datetime.now(timezone.utc) - timedelta(seconds=1))

    def test_enqueue_without_payload_does_not_create_file(self):

        record = _make_record(payload=None)
        self.queue.enqueue(record, backoff_base=60)

        path = self.queue._path_for(record.event_id)
        self._assert_file_not_exists(path)

    def test_get_due_entries_returns_only_past_entries(self):

        record_past = _make_record(entity_id=1)
        record_future = _make_record(entity_id=2)

        self.queue.enqueue(record_past, backoff_base=0)  # due immediately
        self.queue.enqueue(record_future, backoff_base=3600)  # due in 1 hour

        future_entry_path = self.queue._path_for(record_future.event_id)
        with open(future_entry_path, "r", encoding="utf-8") as f:
            future_data = json.load(f)
        far_future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        future_data["next_retry_at"] = far_future
        with open(future_entry_path, "w", encoding="utf-8") as f:
            json.dump(future_data, f, ensure_ascii=False, indent=2)

        due = self.queue.get_due_entries()
        due_event_ids = [e.event_id for e in due]
        self.assertIn(record_past.event_id, due_event_ids)
        self.assertNotIn(record_future.event_id, due_event_ids)

    def test_requeue_increments_count_and_updates_backoff(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        path = self.queue._path_for(record.event_id)

        data = self._read_entry(record.event_id)
        entry = IngestQueueEntry.from_dict(data)

        requeued = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertTrue(requeued)

        updated = self._read_entry(record.event_id)
        self.assertEqual(updated["queue_retry_count"], 1)
        next_retry = datetime.fromisoformat(updated["next_retry_at"])
        expected_lower = datetime.now(timezone.utc) + timedelta(seconds=30)
        expected_upper = datetime.now(timezone.utc) + timedelta(seconds=90)
        self.assertGreaterEqual(next_retry, expected_lower)
        self.assertLessEqual(next_retry, expected_upper)

    def test_requeue_second_time_doubles_backoff(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        data = self._read_entry(record.event_id)
        entry = IngestQueueEntry.from_dict(data)

        self.queue.requeue(entry, max_retries=5, backoff_base=60)
        data2 = self._read_entry(record.event_id)
        entry2 = IngestQueueEntry.from_dict(data2)
        self.queue.requeue(entry2, max_retries=5, backoff_base=60)

        updated = self._read_entry(record.event_id)
        self.assertEqual(updated["queue_retry_count"], 2)
        next_retry = datetime.fromisoformat(updated["next_retry_at"])
        expected_lower = datetime.now(timezone.utc) + timedelta(seconds=90)
        expected_upper = datetime.now(timezone.utc) + timedelta(seconds=150)
        self.assertGreaterEqual(next_retry, expected_lower)
        self.assertLessEqual(next_retry, expected_upper)

    def test_requeue_exceeds_max_moves_to_dead_letter(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        data = self._read_entry(record.event_id)
        data["queue_retry_count"] = 5
        path = self.queue._path_for(record.event_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        entry = IngestQueueEntry.from_dict(data)
        result = self.queue.requeue(entry, max_retries=5, backoff_base=60)

        self.assertFalse(result)
        self._assert_file_not_exists(self.queue._path_for(record.event_id))
        self._assert_file_exists(self.queue._dead_letter_path_for(record.event_id))

    def test_remove_deletes_file(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        self._assert_file_exists(self.queue._path_for(record.event_id))

        self.queue.remove(record.event_id)
        self._assert_file_not_exists(self.queue._path_for(record.event_id))

    def test_get_due_entries_skips_corrupt_files(self):

        corrupt_path = self.queue_path / "corrupt.json"
        corrupt_path.write_text("{ invalid json", encoding="utf-8")

        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 0)

    def test_get_due_entries_malformed_next_retry_at_treated_as_due(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=0)

        path = self.queue._path_for(record.event_id)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["next_retry_at"] = "not-a-date"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        due = self.queue.get_due_entries()
        due_event_ids = [e.event_id for e in due]
        self.assertIn(record.event_id, due_event_ids)

    def test_get_due_entries_malformed_next_retry_at_does_not_strand_other_entries(self):

        good_record = _make_record(entity_id=1)
        bad_record = _make_record(entity_id=2)
        self.queue.enqueue(good_record, backoff_base=0)
        self.queue.enqueue(bad_record, backoff_base=0)

        bad_path = self.queue._path_for(bad_record.event_id)
        with open(bad_path, "r", encoding="utf-8") as f:
            bad_data = json.load(f)
        bad_data["next_retry_at"] = "garbage"
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_data, f, ensure_ascii=False, indent=2)

        due = self.queue.get_due_entries()
        due_event_ids = [e.event_id for e in due]
        self.assertIn(good_record.event_id, due_event_ids)
        self.assertIn(bad_record.event_id, due_event_ids)

    def test_startup_recovery_unlinks_corrupt_processing_files(self):

        corrupt_path = self.queue._processing_path_for("corrupt-entry")
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text("{ invalid json", encoding="utf-8")
        self.assertTrue(corrupt_path.exists())

        new_queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )
        self.assertFalse(corrupt_path.exists())

    def test_startup_recovery_requeues_stranded_processing_entries(self):

        record = _make_record()
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        self.assertTrue(self.queue._processing_path_for(record.event_id).exists())

        new_queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )
        self.assertFalse(new_queue._processing_path_for(record.event_id).exists())
        self.assertTrue(new_queue._path_for(record.event_id).exists())

        due_after = new_queue.get_due_entries()
        self.assertEqual([e.event_id for e in due_after], [record.event_id])

class TestIngestQueueClaiming(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    def _make_record(
        self,
        entity_type: str = "candidate",
        entity_id: int = 1,
    ) -> IngestionRecord:
        return IngestionRecord(
            event_id=str(uuid.uuid4()),
            entity_type=entity_type,  # type: ignore
            entity_id=entity_id,
            status="failed",
            attempt_count=4,
            callback_url="https://example.com/callback",
            error_summary="SomeError: something went wrong",
            callback_delivery_failed=False,
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_get_due_entries_moves_files_to_processing(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].event_id, record.event_id)

        self.assertFalse(self.queue._path_for(record.event_id).exists())
        self.assertTrue(self.queue._processing_path_for(record.event_id).exists())

    def test_claimed_entries_not_returned_by_second_poll(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        first_due = self.queue.get_due_entries()
        self.assertEqual(len(first_due), 1)

        second_due = self.queue.get_due_entries()
        self.assertEqual(len(second_due), 0)

    def test_remove_cleans_processing_file(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        self.queue.get_due_entries()  # claim it

        self.queue.remove(record.event_id)
        self.assertFalse(self.queue._processing_path_for(record.event_id).exists())
        self.assertFalse(self.queue._path_for(record.event_id).exists())

    def test_requeue_moves_entry_back_from_processing_to_queue(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]

        self.assertTrue(self.queue._processing_path_for(entry.event_id).exists())

        requeued = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertTrue(requeued)

        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        self.assertTrue(self.queue._path_for(entry.event_id).exists())
        import json
        with open(self.queue._path_for(entry.event_id), "r") as f:
            data = json.load(f)
        self.assertEqual(data["queue_retry_count"], 1)

    def test_dead_letter_from_processing_state(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]

        entry.queue_retry_count = 5
        result = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertFalse(result)

        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        self.assertTrue(self.queue._dead_letter_path_for(entry.event_id).exists())

    def test_two_poll_cycles_dispatch_once(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        cycle1 = self.queue.get_due_entries()
        self.assertEqual(len(cycle1), 1)

        cycle2 = self.queue.get_due_entries()
        self.assertEqual(len(cycle2), 0)

        self.queue.remove(record.event_id)

        cycle3 = self.queue.get_due_entries()
        self.assertEqual(len(cycle3), 0)

    def test_two_poll_cycles_dispatch_once_then_requeue(self):

        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        cycle1 = self.queue.get_due_entries()
        self.assertEqual(len(cycle1), 1)
        entry = cycle1[0]

        cycle2 = self.queue.get_due_entries()
        self.assertEqual(len(cycle2), 0)

        self.queue.requeue(entry, max_retries=5, backoff_base=0)  # backoff=0 so it's due immediately

        cycle3 = self.queue.get_due_entries()
        self.assertEqual(len(cycle3), 1)
        self.assertEqual(cycle3[0].event_id, record.event_id)

class TestDeadLetterMonitoring(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    def _push_to_dead_letter(self, entity_type: str = "candidate", entity_id: int = 1) -> str:

        record = IngestionRecord(
            event_id=str(uuid.uuid4()),
            entity_type=entity_type,  # type: ignore
            entity_id=entity_id,
            status="failed",
            attempt_count=4,
            callback_url="https://example.com/callback",
            error_summary="SomeError: something went wrong",
            callback_delivery_failed=False,
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]
        entry.queue_retry_count = 5
        self.queue.requeue(entry, max_retries=5, backoff_base=60)
        return record.event_id

    def test_dead_letter_count_returns_zero_when_empty(self):

        self.assertEqual(self.queue.dead_letter_count(), 0)

    def test_dead_letter_count_returns_correct_number(self):

        self._push_to_dead_letter(entity_id=1)
        self._push_to_dead_letter(entity_id=2)
        self.assertEqual(self.queue.dead_letter_count(), 2)

    def test_get_dead_letter_entries_returns_entries(self):

        event_id = self._push_to_dead_letter(entity_id=42)
        entries = self.queue.get_dead_letter_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.event_id, event_id)
        self.assertEqual(entry.entity_type, "candidate")
        self.assertEqual(entry.entity_id, 42)
        self.assertEqual(entry.queue_retry_count, 6)  # incremented by requeue

    def test_get_dead_letter_entries_skips_corrupt_files(self):

        corrupt_path = self.dead_letter_path / "corrupt.json"
        corrupt_path.write_text("{ invalid json", encoding="utf-8")
        entries = self.queue.get_dead_letter_entries()
        self.assertEqual(len(entries), 0)

    def test_clear_dead_letter_removes_single_entry(self):

        id1 = self._push_to_dead_letter(entity_id=1)
        id2 = self._push_to_dead_letter(entity_id=2)
        self.assertEqual(self.queue.dead_letter_count(), 2)

        removed = self.queue.clear_dead_letter(event_id=id1)
        self.assertEqual(removed, 1)
        self.assertEqual(self.queue.dead_letter_count(), 1)
        self.assertTrue(self.queue._dead_letter_path_for(id2).exists())

    def test_clear_dead_letter_removes_all(self):

        self._push_to_dead_letter(entity_id=1)
        self._push_to_dead_letter(entity_id=2)
        self._push_to_dead_letter(entity_id=3)
        self.assertEqual(self.queue.dead_letter_count(), 3)

        removed = self.queue.clear_dead_letter()
        self.assertEqual(removed, 3)
        self.assertEqual(self.queue.dead_letter_count(), 0)

if __name__ == "__main__":
    unittest.main()
