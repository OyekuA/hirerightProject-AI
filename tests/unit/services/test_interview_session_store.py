import pytest
import tempfile
from pathlib import Path

from app.services.interview_session_store import InterviewSessionStore, InterviewSessionRecord


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield InterviewSessionStore(store_path=tmpdir)


class TestInterviewSessionStore:

    def test_create_and_get_by_session_id(self, store: InterviewSessionStore):
        record = store.create(
            candidate_id=1,
            job_id=10,
            rubric=["Communication", "Technical Skills"],
            callback_url="https://example.com/callback",
            bot_id="bot-123",
        )
        assert record.session_id is not None
        assert record.status == "pending"

        fetched = store.get_by_session_id(record.session_id)
        assert fetched is not None
        assert fetched.session_id == record.session_id
        assert fetched.candidate_id == 1
        assert fetched.bot_id == "bot-123"

    def test_create_with_explicit_session_id(self, store: InterviewSessionStore):
        """When session_id is provided explicitly, it must be used instead of generating a new one."""
        record = store.create(
            candidate_id=5,
            job_id=50,
            rubric=["Leadership"],
            callback_url="https://example.com/cb",
            bot_id="bot-explicit",
            session_id="my-custom-session-id",
        )
        assert record.session_id == "my-custom-session-id"
        fetched = store.get_by_session_id("my-custom-session-id")
        assert fetched is not None
        assert fetched.session_id == "my-custom-session-id"
        assert fetched.bot_id == "bot-explicit"

    def test_update(self, store: InterviewSessionStore):
        record = store.create(
            candidate_id=2,
            job_id=20,
            rubric=["Problem Solving"],
            callback_url="https://example.com/cb",
            bot_id="bot-456",
        )
        store.update(record.session_id, status="recording", recording_id="rec-789")
        updated = store.get_by_session_id(record.session_id)
        assert updated is not None
        assert updated.status == "recording"
        assert updated.recording_id == "rec-789"

    def test_get_by_bot_id(self, store: InterviewSessionStore):
        store.create(
            candidate_id=3,
            job_id=30,
            rubric=["A"],
            callback_url="https://ex.com/cb",
            bot_id="bot-find-me",
        )
        found = store.get_by_bot_id("bot-find-me")
        assert found is not None
        assert found.bot_id == "bot-find-me"

    def test_get_by_bot_id_not_found(self, store: InterviewSessionStore):
        found = store.get_by_bot_id("nonexistent")
        assert found is None

    def test_get_all_by_candidate_id(self, store: InterviewSessionStore):
        store.create(candidate_id=99, job_id=1, rubric=["R1"], callback_url="https://ex.com/cb", bot_id="b1")
        store.create(candidate_id=99, job_id=2, rubric=["R2"], callback_url="https://ex.com/cb", bot_id="b2")
        store.create(candidate_id=88, job_id=3, rubric=["R3"], callback_url="https://ex.com/cb", bot_id="b3")

        sessions = store.get_all_by_candidate_id(99)
        assert len(sessions) == 2

        sessions_88 = store.get_all_by_candidate_id(88)
        assert len(sessions_88) == 1

    def test_delete(self, store: InterviewSessionStore):
        record = store.create(
            candidate_id=4,
            job_id=40,
            rubric=["X"],
            callback_url="https://ex.com/cb",
            bot_id="bot-del",
        )
        assert store.get_by_session_id(record.session_id) is not None
        result = store.delete(record.session_id)
        assert result is True
        assert store.get_by_session_id(record.session_id) is None

    def test_delete_not_found(self, store: InterviewSessionStore):
        result = store.delete("nonexistent")
        assert result is False

    def test_get_by_session_id_not_found(self, store: InterviewSessionStore):
        assert store.get_by_session_id("does-not-exist") is None

    def test_update_not_found_raises(self, store: InterviewSessionStore):
        with pytest.raises(KeyError):
            store.update("no-such-id", status="completed")
