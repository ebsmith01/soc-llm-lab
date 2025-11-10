import re

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",   # SSN-like
    r"\b\d{16}\b",              # naive credit card match
]

INJECTION_PATTERNS = [
    "ignore previous",
    "override",
    "disregard",
    "developer mode",
]

def scrub_pii(text: str) -> str:
    out = text
    for pattern in PII_PATTERNS:
        out = re.sub(pattern, "[REDACTED]", out)
    return out

def is_injection(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(p in lowered for p in INJECTION_PATTERNS)
