# evals/run_eval.py

"""
Week 5 — Evaluation Harness

This script:
  1) Loads an eval dataset (baseline.json)
  2) Calls pipeline.answer_query(question)
  3) Computes:
       - exact_match (strict string)
       - semantic_similarity (keyword F1-ish)
       - grounding_score (keyword-based + citations)
  4) Writes a CSV report
  5) Prints aggregate metrics
  6) Optionally computes RAGAS metrics if ragas + datasets are installed

Later you can:
  - Expand eval set (30–100 questions)
  - Slice by type (direct_fact, multi_hop, edge, etc.)
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import sys
from pathlib import Path

# Add project root (soc-llm-lab) to sys.path at runtime
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import pipeline


# -----------------------------
# Paths / config
# -----------------------------

EVAL_PATH = Path(__file__).resolve().parent / "baseline.json"
OUT_CSV = Path(__file__).resolve().parent / "report.csv"

# For optional RAGAS context building
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# -----------------------------
# Helpers for text + metrics
# -----------------------------

def _normalize(s: str) -> str:
    """Very light text normalization for comparisons."""
    return " ".join(s.lower().strip().split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def keyword_f1(pred: str, expected_keywords: List[str]) -> float:
    """
    Tiny surrogate for semantic similarity:
      - Convert expected keywords to a set
      - Count how many appear in the prediction
      - Compute precision/recall/F1 over keyword set

    This is *not* a true embedding similarity,
    but it's a decent cheap signal for this lab.
    """
    if not expected_keywords:
        return 0.0

    pred_norm = _normalize(pred)
    gold_set = set(kw.lower() for kw in expected_keywords)
    hits = {kw for kw in gold_set if kw in pred_norm}

    tp = len(hits)
    fp = len(gold_set - hits)  # missing keywords act like FP-ish
    fn = len(gold_set - hits)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def grounding_score(answer: str, citations: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """
    Very rough grounding proxy:

    - If there are no citations, grounding is 0.0
    - Otherwise, we treat 'answer' as grounded to the extent that
      expected_keywords appear in the answer AND citations exist.

    Later you could:
      - Load the actual chunks via citations and check overlap
      - Integrate RAGAS for hallucination scoring
    """
    if not citations:
        return 0.0

    # Reuse keyword_f1 as a grounding-ish measure:
    return keyword_f1(answer, expected_keywords)


# -----------------------------
# NEW: safe refusal detection
# -----------------------------

SAFE_REFUSAL_PHRASES = [
    "i don't know",
    "i do not know",
    "not in the documents",
    "not covered by the documents",
    "not covered by these documents",
    "not supported by the corpus",
    "cannot answer based on the provided context",
    "cannot answer from the provided context",
    "not enough information in the documents",
]


def is_safe_refusal(answer: str) -> bool:
    """
    Detects when the model is (safely) refusing instead of hallucinating.
    Used to reward edge / safety / out-of-scope questions.
    """
    if not answer:
        return False
    lower = answer.lower()
    return any(p in lower for p in SAFE_REFUSAL_PHRASES)


# -----------------------------
# Optional: load chunk index for RAGAS
# -----------------------------

def load_chunk_index(path: Path) -> Dict[str, str]:
    """
    Build a simple index from chunk id -> text, using chunks.jsonl.

    We try both:
      - record["id"]
      - record["metadata"]["chunk_id"] (if present)

    so RAGAS contexts can be reconstructed from citations.
    """
    index: Dict[str, str] = {}
    if not path.exists():
        return index

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")

            # Primary key: "id"
            rid = rec.get("id")
            if rid and text:
                index[str(rid)] = text

            # Secondary: metadata.chunk_id
            meta = rec.get("metadata") or {}
            cid = meta.get("chunk_id")
            if cid and text:
                index[str(cid)] = text

    return index


# -----------------------------
# Core eval loop
# -----------------------------

def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval() -> None:
    data = load_eval_dataset(EVAL_PATH)
    rows: List[Dict[str, Any]] = []

    n = len(data)
    print(f"Running eval on {n} questions from {EVAL_PATH}...")

    exact_scores: List[float] = []
    semantic_scores: List[float] = []
    grounding_scores: List[float] = []
    latencies: List[float] = []

    # We’ll also collect per-question contexts for optional RAGAS.
    ragas_questions: List[str] = []
    ragas_answers: List[str] = []
    ragas_contexts: List[List[str]] = []

    # Pre-load chunk index (used only if RAGAS is installed)
    chunk_index = load_chunk_index(CHUNKS_PATH)

    for item in data:
        qid = item["id"]
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_keywords = item.get("expected_keywords", [])
        qtype = item.get("type", "")  # NEW: keep question type

        print(f"\n→ [{qid}] {question}")

        t0 = time.time()
        result = pipeline.answer_query(question, top_k=6)
        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)

        answer = result.get("answer", "") or ""
        citations = result.get("citations", []) or []

        em = exact_match(answer, expected_answer)
        sem = keyword_f1(answer, expected_keywords)
        grd = grounding_score(answer, citations, expected_keywords)

        # NEW: reward safe refusals on edge / safety / out-of-scope questions
        if qtype in ("edge", "out_of_scope", "safety") and is_safe_refusal(answer):
            em = 1.0
            sem = 1.0
            grd = 1.0

        exact_scores.append(em)
        semantic_scores.append(sem)
        grounding_scores.append(grd)

        print(f"  latency: {dt_ms:.1f} ms")
        print(f"  exact_match: {em:.2f}, semantic: {sem:.2f}, grounding: {grd:.2f}")

        # Try to reconstruct contexts from citations for RAGAS
        ctx_texts: List[str] = []
        for c in citations:
            # c is whatever pipeline stored (often metadata dict)
            # we try chunk_id, id fields in that dict
            cid = c.get("chunk_id") or c.get("id")
            if cid and cid in chunk_index:
                ctx_texts.append(chunk_index[cid])

        # Fallback: empty contexts if we can't resolve them
        ragas_questions.append(question)
        ragas_answers.append(answer)
        ragas_contexts.append(ctx_texts)

        rows.append(
            {
                "id": qid,
                "type": qtype,  # NEW: keep type on each row
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "expected_keywords": "|".join(expected_keywords),
                "exact_match": em,                # keep as float
                "semantic_keyword_f1": sem,       # keep as float
                "grounding_score": grd,           # keep as float
                "latency_ms": dt_ms,              # keep as float
                "citations": citations,           # keep as raw list; we'll json.dumps later
            }
        )

    # -----------------------------
    # NEW: define win/loss rule
    # -----------------------------

    def is_win(row: Dict[str, Any]) -> bool:
        """
        Core success criterion:
          - semantic_keyword_f1 >= 0.5
          - grounding_score      >= 0.5

        You can tune these thresholds later.
        """
        sem = float(row["semantic_keyword_f1"])
        grd = float(row["grounding_score"])
        return sem >= 0.5 and grd >= 0.5

    for row in rows:
        row["win"] = 1 if is_win(row) else 0

    # Aggregate summary
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    exact_avg = _avg(exact_scores)
    sem_avg = _avg(semantic_scores)
    grd_avg = _avg(grounding_scores)
    lat_avg = _avg(latencies)
    lat_p50 = sorted(latencies)[int(0.5 * (len(latencies) - 1))]
    lat_p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]

    total_wins = sum(row["win"] for row in rows)
    win_rate = total_wins / n if n else 0.0

    print("\n================ EVAL SUMMARY ================")
    print(f"Total questions: {n}")
    print(f"Win rate (semantic F1 + grounding >= 0.5): {total_wins} / {n} = {win_rate:.1%}")
    print(f"Exact match avg:         {exact_avg:.3f}")
    print(f"Semantic keyword F1 avg: {sem_avg:.3f}")
    print(f"Grounding score avg:     {grd_avg:.3f}")
    print(f"Latency avg: {lat_avg:.1f} ms | p50: {lat_p50:.1f} ms | p95: {lat_p95:.1f} ms")

    # Per-type breakdown
    by_type: Dict[str, Dict[str, int]] = {}
    for row in rows:
        t = row.get("type", "unknown") or "unknown"
        by_type.setdefault(t, {"wins": 0, "total": 0})
        by_type[t]["total"] += 1
        by_type[t]["wins"] += row["win"]

    print("\nBreakdown by type:")
    for t, stats in by_type.items():
        tot = stats["total"]
        wins = stats["wins"]
        rate = wins / tot if tot else 0.0
        print(f"  {t:12s}: {wins:3d} / {tot:3d} = {rate:.1%}")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "type",
                "question",
                "answer",
                "expected_answer",
                "expected_keywords",
                "exact_match",
                "semantic_keyword_f1",
                "grounding_score",
                "latency_ms",
                "win",
                "citations",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "type": row.get("type", ""),
                    "question": row["question"],
                    "answer": row["answer"],
                    "expected_answer": row["expected_answer"],
                    "expected_keywords": row["expected_keywords"],
                    "exact_match": f"{row['exact_match']:.3f}",
                    "semantic_keyword_f1": f"{row['semantic_keyword_f1']:.3f}",
                    "grounding_score": f"{row['grounding_score']:.3f}",
                    "latency_ms": f"{row['latency_ms']:.1f}",
                    "win": row["win"],
                    "citations": json.dumps(row["citations"]),
                }
            )

    print(f"\n📊 Detailed report written to: {OUT_CSV}")

    # Optional: RAGAS evaluation
    maybe_run_ragas(ragas_questions, ragas_answers, ragas_contexts)


# -----------------------------
# Optional RAGAS hook
# -----------------------------

def maybe_run_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> None:
    """
    If ragas + datasets are installed, run a small RAGAS eval.

    We'll compute:
      - answer_relevancy
      - faithfulness
      - context_precision
      - context_recall

    If libraries are missing, we just print a message and return.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("\n(RAGAS not installed — skipping RAGAS metrics.)")
        print("To enable: pip install ragas datasets")
        return

    # Build a minimal dataset structure RAGAS expects
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    ds = Dataset.from_dict(data_dict)

    print("\n================ RAGAS METRICS ================")
    result = evaluate(
        ds,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    )
    # result is a Dataset-like object with metric columns
    for metric_name, value in result.items():
        # result items may be lists; take mean if so
        if isinstance(value, list) and value:
            avg = sum(value) / len(value)
            print(f"{metric_name}: {avg:.3f}")
        else:
            print(f"{metric_name}: {value}")


if __name__ == "__main__":
    run_eval()