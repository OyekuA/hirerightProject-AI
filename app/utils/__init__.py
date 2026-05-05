"""Utility functions shared across the application."""

import json
import re
from typing import Union

import hashlib
import structlog
from app.config import get_settings

logger = structlog.get_logger()


def parse_llm_json(generated: str) -> Union[dict, list]:
    """Extract JSON (dict or list) from the LLM's response, tolerating surrounding text.

    Attempts direct JSON parsing first. If that fails, searches for the outermost
    JSON object or array using regex. Logs the raw response for debugging.

    Args:
        generated: Raw text returned by the LLM (may contain markdown fences, extra text).

    Returns:
        Parsed JSON as a dict or list.

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    payload_hash = hashlib.sha256(generated.encode()).hexdigest()[:8]
    logger.debug("Parsing LLM response", length=len(generated), hash=payload_hash)
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
    start = -1
    first_brace = generated.find('{')
    first_bracket = generated.find('[')
    if first_brace == -1 and first_bracket == -1:
        open_char = None
    elif first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        open_char = '{'
        close_char = '}'
        start = first_brace
    else:
        open_char = '['
        close_char = ']'
        start = first_bracket
    if open_char:
        if start != -1:
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(generated)):
                ch = generated[i]
                if escape_next:
                    escape_next = False
                    continue
                if in_string:
                    if ch == '\\':
                        escape_next = True
                    elif ch == '"':
                        in_string = False
                    continue
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == open_char:
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
    payload_hash = hashlib.sha256(generated.encode()).hexdigest()[:8]
    logger.error("Failed to extract JSON from LLM response", length=len(generated), hash=payload_hash)
    raise json.JSONDecodeError("Could not parse JSON from LLM response", "[REDACTED]", 0)