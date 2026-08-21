import json
import pathlib
import re

CANDIDATES_COLLECTION = "candidates"
JOBS_COLLECTION = "jobs"

EMPLOYMENT_TYPES = (
    "full_time",
    "part_time",
    "contract",
    "freelance",
    "internship",
    "temporary",
    "volunteer",
    "apprenticeship",
    "self_employed",
)

WORK_MODES = (
    "remote",
    "hybrid",
    "onsite",
)

_CONFIG_PATH = pathlib.Path(__file__).parent / "config" / "experience_ladder.json"

def _load_experience_config():
    try:
        raw = _CONFIG_PATH.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load experience ladder config at {_CONFIG_PATH}: {e}") from e
    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in experience ladder config at {_CONFIG_PATH}: {e}") from e

    if "ladder" not in data or "aliases" not in data or "keywords" not in data or "numeric_map" not in data:
        raise RuntimeError(f"Experience ladder config missing required keys at {_CONFIG_PATH}")

    ladder = data["ladder"]
    aliases = data["aliases"]
    keywords_raw = data["keywords"]
    numeric_map_raw = data["numeric_map"]
    version = data.get("version", 1)

    if not isinstance(ladder, list) or not all(isinstance(x, str) for x in ladder):
        raise RuntimeError("ladder must be a list of strings")
    if len(ladder) != len(set(ladder)):
        raise RuntimeError("ladder must contain unique values")
    if not isinstance(aliases, dict):
        raise RuntimeError("aliases must be a dict")
    ladder_set = set(ladder)
    for k, v in aliases.items():
        if v not in ladder_set:
            raise RuntimeError(f"alias '{k}' -> '{v}' not in ladder")

    keyword_rules: list[tuple[re.Pattern, str]] = []
    for entry in keywords_raw:
        if not isinstance(entry, dict) or "pattern" not in entry or "canonical" not in entry:
            raise RuntimeError(f"keyword entry invalid: {entry}")
        pat = entry["pattern"]
        canon = entry["canonical"]
        if canon not in ladder_set:
            raise RuntimeError(f"keyword canonical '{canon}' not in ladder")
        try:
            compiled = re.compile(pat)
        except re.error as e:
            raise RuntimeError(f"keyword pattern compile failed for '{pat}': {e}") from e
        keyword_rules.append((compiled, canon))

    numeric_map: dict[int, str] = {}
    for k, v in numeric_map_raw.items():
        try:
            ik = int(k)
        except Exception as e:
            raise RuntimeError(f"numeric_map key '{k}' must be int-convertible: {e}") from e
        if v not in ladder_set:
            raise RuntimeError(f"numeric_map value '{v}' not in ladder")
        numeric_map[ik] = v

    return version, ladder, aliases, keyword_rules, numeric_map

VOCAB_VERSION, EXPERIENCE_LEVEL_LADDER, _EXACT_VARIANTS, _KEYWORD_RULES, _NUMERIC_LEVEL = _load_experience_config()


def get_ladder() -> list[str]:
    return EXPERIENCE_LEVEL_LADDER


def get_aliases() -> dict[str, str]:
    return _EXACT_VARIANTS


def get_keywords() -> list[tuple[re.Pattern, str]]:
    return _KEYWORD_RULES


def get_numeric_map() -> dict[int, str]:
    return _NUMERIC_LEVEL


def get_vocab_version() -> int:
    return VOCAB_VERSION


def canonicalize_experience_level(level: str) -> str:
    cleaned = level.lower().strip().replace("-", " ")

    if cleaned in _EXACT_VARIANTS:
        return _EXACT_VARIANTS[cleaned]

    m = re.match(r"(?:l|level|ic)\s*(\d+)$", cleaned)
    if m:
        n = int(m.group(1))
        return _NUMERIC_LEVEL.get(n, cleaned)

    for pattern, canonical in _KEYWORD_RULES:
        if pattern.search(cleaned):
            return canonical

    return cleaned
