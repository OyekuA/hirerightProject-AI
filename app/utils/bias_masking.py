import copy
import re
from typing import Any


def mask_candidate_for_scoring(candidate_payload: dict) -> dict:
    masked = copy.deepcopy(candidate_payload)
    masked.pop("name", None)
    # skills_vector is a 1536-float matching artifact (~21KB JSON); it must
    # never reach an LLM prompt (blows the prompt cap and breaks scoring).
    masked.pop("skills_vector", None)
    location = masked.get("location")
    if isinstance(location, str) and "," in location:
        masked["location"] = location.strip().rsplit(",", 1)[-1].strip()
    elif isinstance(location, str):
        masked.pop("location", None)
    else:
        masked.pop("location", None)
    return masked


_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def mask_transcript_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masked = []
    for turn in turns:
        text = turn.get("text", "")
        speaker_name = turn.get("speaker", "Unknown")

        text = _NAME_PATTERN.sub("[REDACTED]", text)

        masked.append({
            "speaker": "Candidate",
            "text": text,
        })
    return masked
