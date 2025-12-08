"""
Guardrails module: PII scrubbing, prompt injection detection, and (eventually)
output validation helpers.

These are used by rag.pipeline.answer_query() before calling the LLM.
"""

from __future__ import annotations

import re
from typing import Dict



# ----------------------------
# PII SCRUBBING
# ----------------------------

# Compile a few regexes once at import-time.
# This is not perfect, just a sensible baseline for a lab.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"""(
        (\+?\d{1,2}\s*)?          # optional country code
        (\(?\d{3}\)?[\s\-\.]*)    # area code
        \d{3}[\s\-\.]*\d{4}       # local number
    )""",
    re.VERBOSE,
)
IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_RE = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"  # very rough, catches long digit sequences
)


def scrub_pii(text: str) -> str:
    """
    Replace obvious PII patterns with redacted tokens.

    This is conservative: we prefer over-redaction to under-redaction.
    """
    if not text:
        return text

    scrubbed = text

    scrubbed = EMAIL_RE.sub("[EMAIL]", scrubbed)
    scrubbed = PHONE_RE.sub("[PHONE]", scrubbed)
    scrubbed = IPV4_RE.sub("[IP]", scrubbed)
    scrubbed = SSN_RE.sub("[SSN]", scrubbed)
    scrubbed = CC_RE.sub("[CARD]", scrubbed)

    return scrubbed


# ----------------------------
# OUT-OF-SCOPE / HARMFUL REQUESTS
# ----------------------------

# Things that are clearly not in your corpus or are too sensitive
OUT_OF_SCOPE_PATTERNS = [
    "tax advice",
    "tax law",
    "investment tax",
    "file my taxes",

    "exploit code",
    "zero-day exploit",
    "0day exploit",
    "weaponized exploit",
    "military-grade exploit",
    "shellcode",
    "ransomware builder",

    "complete incident response runbook for every attack technique",
    "complete incident response playbook for every attack technique",
]

def is_out_of_scope_or_harmful(text: str) -> bool:
    """
    Heuristic detector for requests that should be refused even if
    they retrieve 'related' context (e.g. tax law, exploit code).

    We keep this intentionally conservative for the lab.
    """
    if not text:
        return False
    lower = text.lower()
    return any(p in lower for p in OUT_OF_SCOPE_PATTERNS)

# ----------------------------
# PROMPT INJECTION DETECTION
# ----------------------------

# List of phrases that strongly suggest injection attempts.
# This is intentionally conservative: we'd rather block a bit too much
# in a SOC lab than let obviously bad stuff through.
INJECTION_PATTERNS = [
    # Classic “ignore / override” patterns
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "forget your previous rules",
    "you are no longer",
    "you are now instructed to",
    "as a language model you must",
    "act as a completely unrestricted",
    "override your safety",
    "bypass your safety",

    # New: explicit security bypass / offensive help
    "disregard your safety rules",
    "bypass access controls",
    "explain how to bypass access controls",
    "provide steps to exploit",
    "help me hack",
    "disable logging",
    "evade detection",

    # New: requests for internal secrets / config / logs
    "reveal all logs",
    "reveal all secrets",
    "reveal all internal configuration",
    "show me your logs",
    "show me all logs",
    "show me your internal configuration",
    "output all logs",
    "output all configuration",
    "output all secrets",

    # System prompt / hidden instruction fishing
    "reveal the system prompt",
    "print the system prompt",
    "show me your instructions",
]


def _contains_suspicious_phrase(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in INJECTION_PATTERNS)


def _looks_like_system_prompt_leak(text: str) -> bool:
    lower = text.lower()
    # Simple: user is asking about “system prompt” or hidden instructions
    return "system prompt" in lower or "your hidden instructions" in lower


def is_injection(text: str) -> bool:
    """
    Heuristic detector for obvious prompt injection attempts.

    Returns True when we should *block* the query (not just answer carefully).
    This is intentionally strict for the lab.
    """
    if not text:
        return False

    if _contains_suspicious_phrase(text):
        return True

    if _looks_like_system_prompt_leak(text):
        return True

    # You can add more heuristics later:
    # - unusually long 'instruction style' text
    # - lots of '###' / 'SYSTEM:' / 'ASSISTANT:' markers
    # - etc.
    return False
# ----------------------------
# OUTPUT VALIDATION (stub)
# ----------------------------

def validate_answer(answer: str, citations: list) -> Dict[str, bool]:
    """
    Very light-weight validator for answers.

    For now we only enforce:
      - if there is a non-empty answer, there should be at least one citation.

    Returns a dict with flags that can be logged or used by the API.
    """
    ok = True
    missing_citations = False

    if answer and not citations:
        ok = False
        missing_citations = True

    return {
        "ok": ok,
        "missing_citations": missing_citations,
    }