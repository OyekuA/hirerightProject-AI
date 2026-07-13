import json
import pytest
from unittest.mock import MagicMock, patch

from app.services.interview_service import (
    normalize_transcript_turns,
    grade_transcript,
)
from app.utils.bias_masking import mask_transcript_turns


class TestNormalizeTranscriptTurns:

    def test_excludes_host_turns(self):
        raw = [
            {"speaker": {"name": "Host", "label": ""}, "words": [{"text": "Welcome to the interview"}]},
            {"speaker": {"name": "John Doe", "label": ""}, "words": [{"text": "I have 5 years experience"}]},
        ]
        normalized = normalize_transcript_turns(raw)
        assert len(normalized) == 1
        assert normalized[0]["speaker"] == "Candidate"
        assert "experience" in normalized[0]["text"]

    def test_excludes_organizer_label(self):
        raw = [
            {"speaker": {"name": "Alice", "label": "organizer"}, "words": [{"text": "Let's start"}]},
        ]
        normalized = normalize_transcript_turns(raw)
        assert len(normalized) == 0

    def test_masks_candidate_name(self):
        raw = [
            {"speaker": {"name": "Jane Smith", "label": ""}, "words": [{"text": "My name is Jane Smith"}]},
        ]
        normalized = normalize_transcript_turns(raw)
        assert normalized[0]["speaker"] == "Candidate"
        assert "[REDACTED]" in normalized[0]["text"]


class TestIsHostSpeaker:

    def test_is_host_bool_true_wins_over_name(self):
        """Explicit is_host=True marks speaker as host regardless of name/label."""
        from app.services.interview_service import _is_host_speaker
        speaker = {"name": "John Doe", "label": "", "is_host": True}
        assert _is_host_speaker(speaker) is True

    def test_is_host_bool_false_not_excluded(self):
        """Explicit is_host=False keeps the speaker (bool wins over name regex)."""
        from app.services.interview_service import _is_host_speaker
        speaker = {"name": "Host Person", "label": "", "is_host": False}
        assert _is_host_speaker(speaker) is False

    def test_is_host_fallback_name_regex(self):
        """Without is_host key, the name regex catches 'host'/'organizer'."""
        from app.services.interview_service import _is_host_speaker
        assert _is_host_speaker({"name": "Host", "label": ""}) is True
        assert _is_host_speaker({"name": "Organizer", "label": ""}) is True
        assert _is_host_speaker({"name": "Moderator", "label": ""}) is False

    def test_is_host_fallback_label_regex(self):
        """Without is_host key, the label regex catches 'host'/'organizer'."""
        from app.services.interview_service import _is_host_speaker
        assert _is_host_speaker({"name": "Alice", "label": "organizer"}) is True
        assert _is_host_speaker({"name": "Bob", "label": "host"}) is True
        assert _is_host_speaker({"name": "Charlie", "label": "participant"}) is False


class TestGradeTranscript:

    @pytest.mark.asyncio
    async def test_deterministic_overall_score_override(self):
        """Python should compute overall_score = mean of per_criterion_scores."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "per_criterion_scores": {"Communication": 80, "Technical": 90, "Culture": 70},
            "overall_score": 99,  # LLM's value — should be overridden
            "strengths": ["Good communicator"],
            "red_flags": [],
            "recommendation": "hire",
        })

        result = grade_transcript(
            llm=mock_llm,
            rubric=["Communication", "Technical", "Culture"],
            raw_turns=[
                {"speaker": {"name": "Candidate", "label": ""}, "words": [{"text": "I have 10 years exp"}]},
            ],
        )

        # Python computed: round((80+90+70)/3) = round(80) = 80
        assert result["overall_score"] == 80
        assert result["per_criterion_scores"]["Communication"] == 80
        assert result["per_criterion_scores"]["Technical"] == 90
        assert result["recommendation"] == "hire"

    @pytest.mark.asyncio
    async def test_clamps_scores_to_0_100(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "per_criterion_scores": {"A": -10, "B": 150},
            "overall_score": 50,
            "strengths": [],
            "red_flags": [],
            "recommendation": "review",
        })

        result = grade_transcript(
            llm=mock_llm,
            rubric=["A", "B"],
            raw_turns=[{"speaker": {"name": "X"}, "words": [{"text": "hello"}]}],
        )
        assert result["per_criterion_scores"]["A"] == 0
        assert result["per_criterion_scores"]["B"] == 100
        # round((0+100)/2) = 50
        assert result["overall_score"] == 50

    @pytest.mark.asyncio
    async def test_defaults_recommendation_to_review(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "per_criterion_scores": {"X": 50},
            "overall_score": 50,
            "strengths": [],
            "red_flags": [],
            "recommendation": "invalid_value",
        })

        result = grade_transcript(
            llm=mock_llm,
            rubric=["X"],
            raw_turns=[{"speaker": {"name": "X"}, "words": [{"text": "hello"}]}],
        )
        assert result["recommendation"] == "review"
