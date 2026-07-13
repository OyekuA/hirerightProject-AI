import re

CANDIDATES_COLLECTION = "candidates"
JOBS_COLLECTION = "jobs"

EXPERIENCE_LEVEL_LADDER = [
    "intern",
    "apprentice",
    "trainee",
    "junior",
    "associate",
    "mid level",
    "engineer",
    "senior",
    "staff",
    "specialist",
    "lead",
    "principal",
    "distinguished",
    "fellow",
    "manager",
    "senior manager",
    "director",
    "senior director",
    "vice president",
    "senior vice president",
    "executive vice president",
    "chief officer (cto, cio, cdo, etc.)",
    "president",
    "ceo",
]

_EXACT_VARIANTS: dict[str, str] = {
    "jr": "junior",
    "jr.": "junior",
    "entry level": "junior",
    "graduate": "junior",
    "fresh graduate": "junior",
    "mid": "mid level",
    "midlevel": "mid level",
    "mid senior": "senior",
    "sr": "senior",
    "sr.": "senior",
    "senior level": "senior",
    "staff engineer": "staff",
    "lead engineer": "lead",
    "tech lead": "lead",
    "technical lead": "lead",
    "principle": "principal",
    "principle engineer": "principal",
    "engineering manager": "manager",
    "product manager": "manager",
    "managerial": "manager",
    "group manager": "senior manager",
    "director level": "director",
    "head of engineering": "director",
    "head of product": "director",
    "vp": "vice president",
    "vp of engineering": "vice president",
    "vp of product": "vice president",
    "svp": "senior vice president",
    "evp": "executive vice president",
    "c level": "chief officer (cto, cio, cdo, etc.)",
    "cto": "chief officer (cto, cio, cdo, etc.)",
    "cio": "chief officer (cto, cio, cdo, etc.)",
    "cdo": "chief officer (cto, cio, cdo, etc.)",
    "chief technology officer": "chief officer (cto, cio, cdo, etc.)",
    "chief information officer": "chief officer (cto, cio, cdo, etc.)",
    "chief data officer": "chief officer (cto, cio, cdo, etc.)",
    "ceo": "ceo",
    "president": "president",
}

_KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bexecutive vice president\b"), "executive vice president"),
    (re.compile(r"\bsenior vice president\b"),    "senior vice president"),
    (re.compile(r"\bceo\b"),                      "ceo"),
    (re.compile(r"\bpresident\b"),                "president"),
    (re.compile(r"\bexecutive director\b"),       "senior director"),
    (re.compile(r"\bsenior director\b"),          "senior director"),
    (re.compile(r"\bdirector\b"),                 "director"),
    (re.compile(r"\bsenior manager\b"),           "senior manager"),
    (re.compile(r"\bgroup manager\b"),            "senior manager"),
    (re.compile(r"\bmanager\b"),                  "manager"),
    (re.compile(r"\bprincipal\b"),                "principal"),
    (re.compile(r"\bdistinguished\b"),            "distinguished"),
    (re.compile(r"\bfellow\b"),                   "fellow"),
    (re.compile(r"\bstaff\b"),                    "staff"),
    (re.compile(r"\blead\b"),                     "lead"),
    (re.compile(r"\bsenior\b"),                   "senior"),
    (re.compile(r"\bmid\b"),                      "mid level"),
    (re.compile(r"\bjunior\b"),                   "junior"),
    (re.compile(r"\bjr\b"),                       "junior"),
    (re.compile(r"\bentry\b"),                    "junior"),
    (re.compile(r"\bapprentice\b"),               "apprentice"),
    (re.compile(r"\btrainee\b"),                  "trainee"),
    (re.compile(r"\bintern\b"),                   "intern"),
]

_NUMERIC_LEVEL: dict[int, str] = {
    1: "intern",
    2: "junior",
    3: "mid level",
    4: "senior",
    5: "staff",
    6: "principal",
    7: "distinguished",
    8: "fellow",
}


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
