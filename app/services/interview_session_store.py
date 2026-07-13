import json
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
import structlog

_logger = structlog.get_logger(__name__)

SessionStatus = Literal["pending", "recording", "transcribing", "grading", "completed", "failed"]


@dataclass
class InterviewSessionRecord:
    session_id: str
    candidate_id: int
    job_id: int
    rubric: list[str]
    callback_url: str
    bot_id: str
    recording_id: Optional[str] = None
    status: SessionStatus = "pending"
    result: Optional[dict] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterviewSessionRecord":
        return cls(**data)


class InterviewSessionStore:

    def __init__(self, store_path: str):
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        _logger.info("Interview session store initialized", store_path=self._store_path)

    def _path_for(self, session_id: str) -> Path:
        return self._store_path / f"{session_id}.json"

    def create(
        self,
        candidate_id: int,
        job_id: int,
        rubric: list[str],
        callback_url: str,
        bot_id: str,
        session_id: Optional[str] = None,
    ) -> InterviewSessionRecord:
        if session_id is None:
            session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = InterviewSessionRecord(
            session_id=session_id,
            candidate_id=candidate_id,
            job_id=job_id,
            rubric=rubric,
            callback_url=callback_url,
            bot_id=bot_id,
            status="pending",
            recording_id=None,
            result=None,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            with open(self._path_for(session_id), "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

        _logger.debug("Created interview session", session_id=session_id, candidate_id=candidate_id)
        return record

    def update(self, session_id: str, **kwargs) -> None:
        with self._lock:
            path = self._path_for(session_id)
            if not path.exists():
                raise KeyError(f"No interview session with session_id {session_id}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.update(kwargs)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        _logger.debug("Updated interview session", session_id=session_id)

    def get_by_session_id(self, session_id: str) -> Optional[InterviewSessionRecord]:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return InterviewSessionRecord.from_dict(data)

    def get_by_bot_id(self, bot_id: str) -> Optional[InterviewSessionRecord]:
        with self._lock:
            for p in self._store_path.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("bot_id") == bot_id:
                        return InterviewSessionRecord.from_dict(data)
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning("Skipping corrupt session file", file_path=str(p), error=str(e))
        return None

    def get_all_by_candidate_id(self, candidate_id: int) -> List[InterviewSessionRecord]:
        results: list[InterviewSessionRecord] = []
        with self._lock:
            for p in self._store_path.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if int(data.get("candidate_id", -1)) == candidate_id:
                        results.append(InterviewSessionRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning("Skipping corrupt session file", file_path=str(p), error=str(e))
        return results

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
        _logger.debug("Deleted interview session", session_id=session_id)
        return True
