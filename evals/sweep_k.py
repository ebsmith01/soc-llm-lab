"""
Sweep script for (top_k, alpha) retrieval configs.

Usage:
    python -m evals.sweep_k

It will:
  - load evals/baseline.json
  - run all (top_k, alpha) combinations
  - print a leaderboard sorted by accuracy desc, then latency asc.

Eval schema (your current baseline.json):

[
  {
    "id": "q001",
    "question": "...",
    "type": "direct_fact",
    "source_hint": "...",
    "expected_answer": "...",
    "expected_keywords": ["...", "..."]
  },
  ...
]

Scoring uses expected_keywords (or answer_substrings if present).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rag import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "evals" / "baseline.json"

# Hyperparams to sweep
TOP_K_VALUES = [4]
ALPHAS = [0.6]


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class EvalQuestion:
    qid: str
    question: str
    keywords: List[str]


@dataclass
class ConfigResult:
    top_k: int
    alpha: float
    total: int
    correct: int
    accuracy: float
    latencies_ms: List[float]


# -------------------------------------------------------------------
# Eval loading / scoring
# -------------------------------------------------------------------

def load_eval_questions(path: Path = EVAL_PATH) -> List[EvalQuestion]:
    """
    Load baseline eval questions.

    Supports both shapes:

    1) Your current schema:

    {
      "id": "q001",
      "question": "...",
      "expected_answer": "...",
      "expected_keywords": ["...", "..."]
    }

    2) Older schema (for compatibility):

    {
      "id": "q001",
      "question": "...",
      "answer_substrings": ["...", "..."]
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    questions: List[EvalQuestion] = []
    for item in raw:
        # Prefer your "expected_keywords", fall back to "answer_substrings" if present
        kws = item.get("expected_keywords") or item.get("answer_substrings") or []
        questions.append(
            EvalQuestion(
                qid=item["id"],
                question=item["question"],
                keywords=kws,
            )
        )
    return questions


def answer_matches(answer: str, expected_keywords: List[str]) -> bool:
    """
    Very simple scoring: if ANY expected keyword/phrase appears
    (case-insensitive) in the answer, it counts as correct.

    You can tighten this later (require all keywords, use regex, etc.).
    """
    if not expected_keywords:
        return False

    lower = answer.lower()
    return any(kw.lower() in lower for kw in expected_keywords)


# -------------------------------------------------------------------
# Single config run: (top_k, alpha)
# -------------------------------------------------------------------

def run_single_config(
    top_k: int,
    alpha: float,
    baseline: List[EvalQuestion],
) -> ConfigResult:
    """
    Run one (top_k, alpha) configuration against the full eval set.

    Calls pipeline.answer_query(question, top_k=..., alpha=...)
    and tracks:
      - accuracy (keyword match)
      - per-question latency in ms
    """
    latencies: List[float] = []
    total = 0
    correct = 0

    print(f"\n=== Running config: top_k={top_k}, alpha={alpha:.2f} ===")

    for q in baseline:
        total += 1
        print(f"→ [{q.qid}] {q.question}")

        start = time.perf_counter()
        result: Dict[str, Any] = pipeline.answer_query(
            q.question,
            top_k=top_k,
            alpha=alpha,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        answer = result.get("answer", "") or ""

        is_correct = answer_matches(answer, q.keywords)
        if is_correct:
            correct += 1
            print(f"   ✔ correct (latency={elapsed_ms:.1f} ms)")
        else:
            print(f"   ✖ incorrect (latency={elapsed_ms:.1f} ms)")

    accuracy = (correct / total) * 100.0 if total else 0.0

    return ConfigResult(
        top_k=top_k,
        alpha=alpha,
        total=total,
        correct=correct,
        accuracy=accuracy,
        latencies_ms=latencies,
    )


# -------------------------------------------------------------------
# Latency summary
# -------------------------------------------------------------------

def summarize_latency(latencies_ms: List[float]) -> Tuple[float, float, float]:
    """
    Return avg, p50, p95 latencies in ms.
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0

    avg = sum(latencies_ms) / len(latencies_ms)
    p50 = statistics.median(latencies_ms)

    sorted_lats = sorted(latencies_ms)
    idx_95 = max(0, int(len(sorted_lats) * 0.95) - 1)
    p95 = sorted_lats[idx_95]

    return avg, p50, p95


# -------------------------------------------------------------------
# Top-level sweep
# -------------------------------------------------------------------

def run_sweep():
    # Demo note: quick hyperparameter sweep that prints a sorted leaderboard.
    baseline = load_eval_questions(EVAL_PATH)

    all_results: List[ConfigResult] = []

    for alpha in ALPHAS:
        for top_k in TOP_K_VALUES:
            cfg_result = run_single_config(top_k=top_k, alpha=alpha, baseline=baseline)
            all_results.append(cfg_result)

    # Sort by accuracy desc, then avg latency asc
    def sort_key(r: ConfigResult):
        avg, _, _ = summarize_latency(r.latencies_ms)
        return (-r.accuracy, avg)

    all_results.sort(key=sort_key)

    print("\n\n================ LEADERBOARD ================")
    print("top_k | alpha | accuracy | avg_ms | p50_ms | p95_ms")
    print("------+-------+----------+--------+--------+--------")

    for r in all_results:
        avg, p50, p95 = summarize_latency(r.latencies_ms)
        print(
            f"{r.top_k:5d} | {r.alpha:5.2f} | {r.accuracy:7.1f}% | "
            f"{avg:6.1f} | {p50:6.1f} | {p95:6.1f}"
        )


if __name__ == "__main__":
    run_sweep()
