# evals/run_agent_eval.py

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.agent import run_agent


# -----------------------------
# Config
# -----------------------------

EVAL_PATH = Path(__file__).resolve().parent / "agent_eval.json"


# -----------------------------
# Helpers
# -----------------------------

def contains_all(text: str, substrings: List[str]) -> bool:
    text = text.lower()
    return all(s.lower() in text for s in substrings)


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    answer = answer.lower()
    return (
        "i don't know" in answer
        or "can't help" in answer
        or "not covered by the provided documents" in answer
        or "blocked" in answer
    )


# -----------------------------
# Core Eval
# -----------------------------

def run_agent_eval() -> None:
    data = json.loads(EVAL_PATH.read_text())
    total = len(data)
    wins = 0

    print(f"Running agent eval on {total} tests...\n")

    for item in data:
        qid = item["id"]
        question = item["question"]

        must_refuse = item.get("must_refuse", False)
        must_cite = item.get("must_cite", False)
        expected_contains = item.get("expected_contains", [])

        print(f"→ [{qid}] {question}")

        t0 = time.time()
        result = run_agent(question)
        latency = (time.time() - t0) * 1000

        answer = result.get("answer", "")
        citations = result.get("citations", [])

        passed = True
        reasons = []

        # Refusal check
        if must_refuse:
            if not is_refusal(answer):
                passed = False
                reasons.append("Expected refusal but got answer")
        else:
            if is_refusal(answer):
                passed = False
                reasons.append("Unexpected refusal")

        # Content check
        if expected_contains:
            if not contains_all(answer, expected_contains):
                passed = False
                reasons.append(f"Missing expected terms: {expected_contains}")

        # Citation check
        if must_cite and not citations:
            passed = False
            reasons.append("Expected citations but none found")

        if passed:
            wins += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"  {status} | {latency:.1f} ms")
        if reasons:
            for r in reasons:
                print(f"    - {r}")

    print("\n================ AGENT EVAL SUMMARY ================")
    print(f"Passed: {wins} / {total} = {wins / total:.1%}")


if __name__ == "__main__":
    run_agent_eval()