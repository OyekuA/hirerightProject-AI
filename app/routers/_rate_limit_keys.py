"""Shared key‑extractor functions for per‑entity rate limiting.

These functions are used as `key_func` arguments in `@limiter.limit(...)` decorators
to enforce per‑candidate, per‑job, or per‑target rate limits.
"""

import hashlib
import json
from typing import Optional, Union

from fastapi import Request
from slowapi.util import get_remote_address


def _fingerprint(request: Request, entity_id: Optional[str] = None) -> str:
    """Return a SHA‑256 hex digest of the API key (or client IP) optionally concatenated with entity_id."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        base = api_key.encode()
    else:
        base = get_remote_address(request).encode()
    if entity_id is not None:
        base += str(entity_id).encode()
    return hashlib.sha256(base).hexdigest()


def _read_body(request: Request) -> Union[dict, None]:
    """Synchronously read the cached JSON body from a Starlette Request.

    Starlette caches the raw body bytes in ``request._body`` after the first
    ``await request.body()`` (which FastAPI runs during Pydantic model binding).
    This helper returns the parsed dict, or *None* if the body is absent,
    empty, or not valid JSON.
    """
    try:
        raw = request._body
    except AttributeError:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def candidate_id_key(request: Request) -> str:
    """Extract candidate_id from request body for rate limiting.

    Returns a key of the form "candidate:{fingerprint}" if candidate_id is present
    in the JSON request body; otherwise returns "unknown:{fingerprint}" where the
    fingerprint is derived from the API key (or client IP) alone.
    """
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    candidate_id = body.get("candidate_id")
    if candidate_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"candidate:{_fingerprint(request, entity_id=str(candidate_id))}"


def job_id_key(request: Request) -> str:
    """Extract job_id from request body for rate limiting.

    Returns a key of the form "job:{fingerprint}" if job_id is present
    in the JSON request body; otherwise returns "unknown:{fingerprint}" where the
    fingerprint is derived from the API key (or client IP) alone.
    """
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    job_id = body.get("job_id")
    if job_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"job:{_fingerprint(request, entity_id=str(job_id))}"


def target_id_key(request: Request) -> str:
    """Extract target_id from request body for rate limiting.

    Returns a key of the form "target:{fingerprint}" if target_id is present
    in the JSON request body; otherwise returns "unknown:{fingerprint}" where the
    fingerprint is derived from the API key (or client IP) alone.
    """
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    target_id = body.get("target_id")
    if target_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"target:{_fingerprint(request, entity_id=str(target_id))}"


def candidate_or_job_id_key(request: Request) -> str:
    """Extract a stable identifier from request body for rate limiting.

    Looks for a candidate_id inside a "candidate_context" object first,
    then for a job_id inside a "job_context" object.

    Returns a key of the form "candidate:{fingerprint}" or "job:{fingerprint}"
    if either is found; otherwise returns "unknown:{fingerprint}" where the
    fingerprint is derived from the API key (or client IP) alone.
    """
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    candidate_context = body.get("candidate_context")
    if isinstance(candidate_context, dict) and candidate_context.get("candidate_id") is not None:
        return f"candidate:{_fingerprint(request, entity_id=str(candidate_context['candidate_id']))}"
    job_context = body.get("job_context")
    if isinstance(job_context, dict) and job_context.get("job_id") is not None:
        return f"job:{_fingerprint(request, entity_id=str(job_context['job_id']))}"
    return f"unknown:{_fingerprint(request)}"
