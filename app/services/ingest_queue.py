"""File-based ingest queue for deferred retry of failed ingestion jobs.

Each failed ingestion is persisted as a JSON file in a configurable queue
directory. A background worker polls for due entries and re-dispatches them.
After exceeding the maximum retry count, entries are moved to a dead letter
directory for manual investigation.

An atomic claim step prevents the same entry from being dispatched by
concurrent poll cycles: due files are renamed to a processing sub‑directory
before being returned to the caller.
"""

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

import structlog

from app.services.ingestion_store import IngestionRecord, EntityType

_logger = structlog.get_logger(__name__)


@dataclass
class IngestQueueEntry:
    """A single entry in the ingest queue."""

    event_id: str
    entity_type: EntityType
    entity_id: int
    callback_url: str
    payload: dict
    queue_retry_count: int = 0
    next_retry_at: str = ""
    enqueued_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "callback_url": self.callback_url,
            "payload": self.payload,
            "queue_retry_count": self.queue_retry_count,
            "next_retry_at": self.next_retry_at,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestQueueEntry":
        return cls(**data)


class IngestQueue:
    """Thread‑safe file‑based queue for deferred ingestion retries."""

    def __init__(self, queue_path: str, dead_letter_path: str):
        self._queue_path = Path(queue_path)
        self._dead_letter_path = Path(dead_letter_path)
        self._processing_path = self._queue_path / ".processing"
        self._queue_path.mkdir(parents=True, exist_ok=True)
        self._dead_letter_path.mkdir(parents=True, exist_ok=True)
        self._processing_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        _logger.info(
            "Ingest queue initialized",
            queue_path=str(self._queue_path),
            processing_path=str(self._processing_path),
            dead_letter_path=str(self._dead_letter_path),
        )

    def _path_for(self, event_id: str) -> Path:
        return self._queue_path / f"{event_id}.json"

    def _processing_path_for(self, event_id: str) -> Path:
        return self._processing_path / f"{event_id}.json"

    def _dead_letter_path_for(self, event_id: str) -> Path:
        return self._dead_letter_path / f"{event_id}.json"

    def enqueue(self, record: IngestionRecord, backoff_base: int) -> None:
        """Enqueue a failed ingestion record for deferred retry.

        If the record has no payload, a warning is logged and the method
        returns without writing — the entry cannot be re-run without it.
        """
        if record.payload is None:
            _logger.warning(
                "Cannot enqueue ingestion record without payload",
                event_id=record.event_id,
            )
            return

        now = datetime.now(timezone.utc)
        next_retry_at = now + timedelta(seconds=backoff_base)
        entry = IngestQueueEntry(
            event_id=record.event_id,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            callback_url=record.callback_url,
            payload=record.payload,
            queue_retry_count=0,
            next_retry_at=next_retry_at.isoformat(),
            enqueued_at=now.isoformat(),
        )

        with self._lock:
            with open(self._path_for(entry.event_id), "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

        _logger.info(
            "Enqueued ingestion for retry",
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            next_retry_at=entry.next_retry_at,
        )

    def get_due_entries(self) -> List[IngestQueueEntry]:
        """Atomically claim and return all queue entries whose retry timestamp has passed.

        Each due file is renamed to the processing sub‑directory under the queue
        lock so that subsequent poll cycles will not see it again until it is
        explicitly requeued or removed.
        """
        now = datetime.now(timezone.utc)
        due: List[IngestQueueEntry] = []

        with self._lock:
            for p in sorted(self._queue_path.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entry = IngestQueueEntry.from_dict(data)
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning(
                        "Skipping corrupt queue file",
                        file_path=str(p),
                        error=str(e),
                    )
                    continue

                if datetime.fromisoformat(entry.next_retry_at) <= now:
                    # Atomically claim the entry by moving it to the processing dir
                    processing_path = self._processing_path_for(entry.event_id)
                    try:
                        os.rename(str(p), str(processing_path))
                    except OSError as e:
                        _logger.error(
                            "Failed to claim queue entry",
                            event_id=entry.event_id,
                            file_path=str(p),
                            error=str(e),
                        )
                        continue
                    due.append(entry)

        return due

    def requeue(
        self, entry: IngestQueueEntry, max_retries: int, backoff_base: int
    ) -> bool:
        """Increment retry count and update next_retry_at with exponential backoff.

        Writes the updated entry back to the main queue directory and removes
        the processing copy so the entry becomes visible for the next poll cycle.

        Returns True if the entry was re-queued, False if it exceeded max_retries
        and was moved to the dead letter directory.
        """
        entry.queue_retry_count += 1

        if entry.queue_retry_count > max_retries:
            self.move_to_dead_letter(entry)
            return False

        now = datetime.now(timezone.utc)
        backoff = backoff_base * (2 ** (entry.queue_retry_count - 1))
        entry.next_retry_at = (now + timedelta(seconds=backoff)).isoformat()

        with self._lock:
            # Write updated data back to the main queue path
            with open(self._path_for(entry.event_id), "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            # Remove the processing copy (if any)
            proc_path = self._processing_path_for(entry.event_id)
            if proc_path.exists():
                proc_path.unlink()

        _logger.info(
            "Re-queued ingestion entry",
            event_id=entry.event_id,
            queue_retry_count=entry.queue_retry_count,
            next_retry_at=entry.next_retry_at,
        )
        return True

    def remove(self, event_id: str) -> None:
        """Remove a queue entry by event_id.

        Checks both the main queue and processing directories so that entries
        claimed by ``get_due_entries`` can be removed on success.
        """
        with self._lock:
            for path in (self._path_for(event_id), self._processing_path_for(event_id)):
                if path.exists():
                    path.unlink()
        _logger.debug("Removed queue entry", event_id=event_id)

    def move_to_dead_letter(self, entry: IngestQueueEntry) -> None:
        """Move an entry to the dead letter directory after exhausting retries."""
        with self._lock:
            dl_path = self._dead_letter_path_for(entry.event_id)
            with open(dl_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            # Remove from either queue or processing directory
            for path in (self._path_for(entry.event_id), self._processing_path_for(entry.event_id)):
                if path.exists():
                    path.unlink()

        _logger.error(
            "Ingestion entry moved to dead letter (max retries exceeded)",
            event_id=entry.event_id,
            entity_type=entry.entity_type,
            entity_id=entry.entity_id,
            queue_retry_count=entry.queue_retry_count,
        )

    def dead_letter_count(self) -> int:
        with self._lock:
            return len(list(self._dead_letter_path.glob("*.json")))

    def get_dead_letter_entries(self) -> List[IngestQueueEntry]:
        entries: List[IngestQueueEntry] = []
        with self._lock:
            for p in sorted(self._dead_letter_path.glob("*.json")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entries.append(IngestQueueEntry.from_dict(data))
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning(
                        "Skipping corrupt dead letter file",
                        file_path=str(p),
                        error=str(e),
                    )
        return entries

    def clear_dead_letter(self, event_id: Optional[str] = None) -> int:
        removed = 0
        with self._lock:
            if event_id is not None:
                path = self._dead_letter_path_for(event_id)
                if path.exists():
                    path.unlink()
                    removed = 1
            else:
                for p in self._dead_letter_path.glob("*.json"):
                    p.unlink()
                    removed += 1
        return removed
