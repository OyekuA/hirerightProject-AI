import json
import re
import statistics
from typing import Optional
import structlog

from app.clients.llm import LLMClient, LLMUnavailableError
from app.utils import parse_llm_json
from app.utils.bias_masking import mask_transcript_turns
from app.prompts import TRANSCRIPT_SCORING_PROMPT_TEMPLATE

logger = structlog.get_logger()


def normalize_transcript_turns(
    raw_turns: list[dict],
) -> list[dict]:
    participant_turns: list[dict] = []
    for turn in raw_turns:
        words = turn.get("words", [])
        participant = turn.get("participant", turn.get("speaker", {}))
        is_host = _is_host_speaker(participant)
        text = " ".join(w.get("text", "") for w in words if w.get("text"))
        speaker_name = participant.get("name", "Unknown")

        if is_host:
            continue

        participant_turns.append({
            "speaker": speaker_name,
            "text": text,
        })

    masked = mask_transcript_turns(participant_turns)
    return masked


def _is_host_speaker(speaker: dict) -> bool:
    is_host = speaker.get("is_host")
    if isinstance(is_host, bool):
        return is_host
    name = (speaker.get("name") or "").lower()
    label = (speaker.get("label") or "").lower()
    return bool(re.search(r"\b(host|organizer)\b", name) or re.search(r"\b(host|organizer)\b", label))


def grade_transcript(
    llm: LLMClient,
    rubric: list[str],
    raw_turns: list[dict],
) -> dict:
    normalized = normalize_transcript_turns(raw_turns)

    rubric_json = json.dumps(rubric, ensure_ascii=False)
    transcript_turns_json = json.dumps(normalized, ensure_ascii=False)

    prompt = TRANSCRIPT_SCORING_PROMPT_TEMPLATE.format(
        rubric_json=rubric_json,
        transcript_turns_json=transcript_turns_json,
    )

    generated = llm.generate(prompt, temperature=0)

    try:
        result = parse_llm_json(generated)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON for transcript grading")
        raise LLMUnavailableError("LLM returned unparseable JSON for transcript grading")

    if not isinstance(result, dict):
        raise LLMUnavailableError("LLM returned a non-dict payload for transcript grading")

    per_criterion = result.get("per_criterion_scores")
    if not isinstance(per_criterion, dict) or not per_criterion:
        raise LLMUnavailableError("LLM response missing 'per_criterion_scores'")

    for k, v in per_criterion.items():
        try:
            per_criterion[k] = max(0, min(100, int(v)))
        except (TypeError, ValueError):
            per_criterion[k] = 0

    overall_score = round(statistics.mean(per_criterion.values()))
    overall_score = max(0, min(100, overall_score))

    strengths = result.get("strengths", [])
    if not isinstance(strengths, list):
        strengths = []

    red_flags = result.get("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []

    recommendation = result.get("recommendation", "review")
    if recommendation not in ("hire", "no_hire", "review", "strong_hire"):
        recommendation = "review"

    return {
        "per_criterion_scores": per_criterion,
        "overall_score": overall_score,
        "strengths": strengths,
        "red_flags": red_flags,
        "recommendation": recommendation,
    }
