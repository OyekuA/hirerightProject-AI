"""Utility functions shared across the application."""

import json
import re
from typing import Union

import structlog
from app.config import get_settings

logger = structlog.get_logger()


def truncate_to_prompt_cap(text: str) -> str:
    """Silently truncate a text to the maximum allowed prompt length.

    The maximum length is read from the application settings (MAX_PROMPT_CHARS).
    If the text is longer than the limit, it is truncated without raising an
    exception and without logging. Shorter texts are returned unchanged.

    Args:
        text: The input text that may be passed to Gemini.

    Returns:
        The same text if within the limit, otherwise a truncated version
        containing the first MAX_PROMPT_CHARS characters.
    """
    settings = get_settings()
    max_chars = settings.MAX_PROMPT_CHARS
    return text[:max_chars]


def parse_gemini_json(generated: str) -> Union[dict, list]:
    """Extract JSON (dict or list) from Gemini's response, tolerating surrounding text.

    Attempts direct JSON parsing first. If that fails, searches for the outermost
    JSON object or array using regex. Logs the raw response for debugging.

    Args:
        generated: Raw text returned by Gemini (may contain markdown fences, extra text).

    Returns:
        Parsed JSON as a dict or list.

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    logger.debug("Raw Gemini response", raw=generated[:200])
    generated = generated.strip()
    if generated.startswith('```json'):
        generated = generated[7:].lstrip()
    if generated.startswith('```'):
        generated = generated[3:].lstrip()
    if generated.endswith('```'):
        generated = generated[:-3].rstrip()
    generated = generated.strip()
    try:
        return json.loads(generated)
    except json.JSONDecodeError:
        pass
    open_char = None
    close_char = None
    if '{' in generated:
        open_char = '{'
        close_char = '}'
    elif '[' in generated:
        open_char = '['
        close_char = ']'
    if open_char:
        start = generated.find(open_char)
        if start != -1:
            depth = 0
            for i in range(start, len(generated)):
                ch = generated[i]
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        candidate = generated[start:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
    pattern = r'(\{.*\}|\[.*\])'
    matches = re.findall(pattern, generated, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    logger.error("Failed to extract JSON from Gemini response", raw=generated[:500])
    raise json.JSONDecodeError("Could not parse JSON from Gemini response", generated, 0)
