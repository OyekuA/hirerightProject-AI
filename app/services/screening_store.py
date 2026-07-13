import json
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
import structlog

_logger = structlog.get_logger(__name__)

ScreeningStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class ScreeningBatchRecord:
    batch_id: str
    status: ScreeningStatus
    total: int
    job_ref: dict
    results: List[dict]
    callback_url: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScreeningBatchRecord":
        return cls(**data)


class BatchScreeningStore:

    def __init__(self, store_path: str):
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        _logger.info("Batch screening store initialized", store_path=self._store_path)

    def _path_for(self, batch_id: str) -> Path:
        return self._store_path / f"{batch_id}.json"

    def create(
        self,
        total: int,
        job_ref: dict,
        callback_url: Optional[str] = None,
    ) -> ScreeningBatchRecord:
        batch_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = ScreeningBatchRecord(
            batch_id=batch_id,
            status="pending",
            total=total,
            job_ref=job_ref,
            results=[],
            callback_url=callback_url,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            with open(self._path_for(batch_id), "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

        _logger.debug("Created screening batch record", batch_id=batch_id, total=total)
        return record

    def update(self, batch_id: str, **kwargs) -> None:
        with self._lock:
            path = self._path_for(batch_id)
            if not path.exists():
                raise KeyError(f"No screening batch record with batch_id {batch_id}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.update(kwargs)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        _logger.debug("Updated screening batch record", batch_id=batch_id)

    def append_result(self, batch_id: str, result: dict) -> None:
        with self._lock:
            path = self._path_for(batch_id)
            if not path.exists():
                raise KeyError(f"No screening batch record with batch_id {batch_id}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.setdefault("results", [])
            results.append(result)
            data["results"] = results
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        _logger.debug("Appended result to screening batch", batch_id=batch_id)

    def get_by_batch_id(self, batch_id: str) -> Optional[ScreeningBatchRecord]:
        path = self._path_for(batch_id)
        with self._lock:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return ScreeningBatchRecord.from_dict(data)
