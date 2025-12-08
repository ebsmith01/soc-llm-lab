from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import tiktoken


# -------------------------------
# Model pricing (per 1K tokens)
# Adjust if OpenAI pricing changes.
# -------------------------------
COST_TABLE: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4.1": {"input": 0.00500, "output": 0.01500},
}


# -------------------------------
# Token counting
# -------------------------------
def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Return the number of tokens for `text` for the given model.

    This is useful for:
    - Understanding context window usage
    - Estimating cost
    - Designing chunk sizes for RAG

    Example:
        n = token_count(alert_text, model="gpt-4o-mini")
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# -------------------------------
# Cost estimation
# -------------------------------
def estimate_cost(
    model: str,
    input_text: str,
    expected_output_tokens: int = 200,
) -> float:
    """
    Estimate the dollar cost of sending `input_text` to `model` and
    receiving ~expected_output_tokens back.

    This is an approximation for planning & comparison; actual cost may vary.

    Example:
        cost = estimate_cost("gpt-4o-mini", alert_text, expected_output_tokens=300)
    """
    if model not in COST_TABLE:
        raise ValueError(f"Unknown model for cost estimation: {model!r}")

    input_tokens = token_count(input_text, model=model)
    in_cost = (input_tokens / 1000) * COST_TABLE[model]["input"]
    out_cost = (expected_output_tokens / 1000) * COST_TABLE[model]["output"]
    return in_cost + out_cost


# -------------------------------
# JSON validation
# -------------------------------
def check_json(text: str) -> Tuple[bool, Any, str | None]:
    """
    Try to parse `text` as JSON.

    Returns:
        (ok, parsed, error_message)

    - ok: True if json.loads() succeeded, False otherwise
    - parsed: the parsed Python object (dict/list/etc) if ok, else None
    - error_message: error string if failed, else None

    Example:
        ok, data, err = check_json(model_output)
        if not ok:
            log the raw output & error for debugging
    """
    try:
        parsed = json.loads(text)
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, str(e)
