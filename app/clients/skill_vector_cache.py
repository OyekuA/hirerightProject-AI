import re
import threading
from typing import Optional, List

from app.config import get_settings

_CACHE: dict[str, List[float]] = {}
_LOCK = threading.RLock()
_VERSION: Optional[tuple] = None

_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_WS_RE = re.compile(r"\s+")


def normalize_skill_key(skill: str) -> str:
    s = skill.lower().strip()
    s = _PAREN_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _check_version():
    global _VERSION
    settings = get_settings()
    cur = (settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS)
    if _VERSION is None:
        _VERSION = cur
    elif _VERSION != cur:
        with _LOCK:
            if _VERSION != cur:
                _CACHE.clear()
                _VERSION = cur


def lookup(skill: str) -> Optional[List[float]]:
    key = normalize_skill_key(skill)
    with _LOCK:
        return _CACHE.get(key)


def get_or_embed(skill: str, llm) -> Optional[List[float]]:
    try:
        _check_version()
    except Exception:
        pass
    key = normalize_skill_key(skill)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        # compute-if-absent under lock
        try:
            vec = llm.embed(skill)
        except Exception:
            return None
        if vec is None:
            return None
        _CACHE[key] = vec
        return vec


def warm(skills: List[str], llm) -> int:
    try:
        _check_version()
    except Exception:
        pass
    if not skills:
        return 0
    norm_to_orig: dict[str, str] = {}
    for s in skills:
        if s is None:
            continue
        s_str = str(s).strip()
        if not s_str:
            continue
        key = normalize_skill_key(s_str)
        if not key:
            continue
        if key not in norm_to_orig:
            norm_to_orig[key] = s_str

    embedded = 0
    for key, orig in norm_to_orig.items():
        with _LOCK:
            if key in _CACHE:
                continue
        # not cached, try to embed
        vec = get_or_embed(orig, llm)
        if vec is not None:
            embedded += 1
    return embedded


def clear():
    with _LOCK:
        _CACHE.clear()


def cache_size() -> int:
    with _LOCK:
        return len(_CACHE)
