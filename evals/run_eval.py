# evals/run_eval.py

"""
Week 5 — Evaluation Harness

This script:
  1) Loads an eval dataset (baseline.json)
  2) Calls pipeline.answer_query(question)
  3) Computes:
       - exact_match (strict string)
       - semantic_keyword_f1 (keyword F1-ish)
       - grounding_score (citations present + keyword overlap)
  4) Writes a CSV report
  5) Prints aggregate metrics
  6) Optionally computes RAGAS metrics if ragas + datasets are installed
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import os
from datetime import datetime


# -----------------------------
# Add project root to sys.path
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import pipeline  # noqa: E402


# -----------------------------
# Defaults / paths
# -----------------------------

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent / "baseline.json"
DEFAULT_OUT_CSV = Path(__file__).resolve().parent / "report.csv"

# For optional RAGAS context building
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

# FINAL Week 5 tuning defaults
DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 4


# -----------------------------
# Helpers for text + metrics
# -----------------------------

def _normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) and gold else 0.0


def keyword_f1(pred: str, expected_keywords: List[str]) -> float:
    """
    Keyword F1 (cheap semantic proxy)

    We treat expected_keywords as "gold set".
    A keyword counts as present if it appears as a substring in pred_norm.

    Precision = hits / predicted_terms
      - We don't have a true predicted keyword set, so we approximate:
        predicted_terms = hits + "other terms" is unknown.
      - For this harness, we use recall-only style F1:
        precision := recall := hits / |gold|
        F1 := hits/|gold|
      This keeps behavior stable and avoids penalizing verbosity.

    If you want a stricter metric later, switch to token-level sets.
    """
    if not expected_keywords:
        return 0.0

    pred_norm = _normalize(pred)
    gold = [kw.lower().strip() for kw in expected_keywords if kw and kw.strip()]
    if not gold:
        return 0.0

    hits = 0
    for kw in set(gold):
        if kw in pred_norm:
            hits += 1

    # This behaves like recall; in this harness it's "good enough".
    return hits / len(set(gold))


def grounding_score(answer: str, citations: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    """
    Grounding proxy:
      - If no citations -> 0.0
      - Else -> keyword_f1(answer, expected_keywords)

    (So grounding is "semantic signal, but only if citations exist".)
    """
    if not citations:
        return 0.0
    return keyword_f1(answer, expected_keywords)


# -----------------------------
# Safe refusal detection
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
    "i can't do that",
    "i cannot do that",
    "unsafe",
    "i will not provide",
]


def is_safe_refusal(answer: str) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(p in lower for p in SAFE_REFUSAL_PHRASES)


EDGE_TYPES = {"edge", "out_of_scope", "safety"}


# -----------------------------
# Optional: load chunk index
# -----------------------------

def load_chunk_index(path: Path) -> Dict[str, str]:
    """
    Build index from:
      - record["id"] -> record["text"]
      - record["metadata"]["chunk_id"] -> record["text"]  (if present)
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
            text = rec.get("text", "") or ""
            if not text:
                continue

            rid = rec.get("id")
            if rid is not None:
                index[str(rid)] = text

            meta = rec.get("metadata") or {}
            cid = meta.get("chunk_id")
            if cid is not None:
                index[str(cid)] = text

    return index


def resolve_contexts_from_citations(
    citations: List[Dict[str, Any]],
    chunk_index: Dict[str, str],
    max_contexts: int = 2,
) -> List[str]:
    """
    Your pipeline citations look like:
      {
        "id": ...,
        "source": ...,
        "page_num": ...,
        "score": ...,
        "metadata": { "chunk_id": "...", "doc_id": "...", ... }
      }

    We try:
      - citation["metadata"]["chunk_id"]
      - citation["id"]
      - citation["metadata"]["id"] (just in case)
    """
    ctx: List[str] = []
    for c in citations or []:
        meta = c.get("metadata") or {}
        candidates = [
            meta.get("chunk_id"),
            c.get("id"),
            meta.get("id"),
        ]
        for key in candidates:
            if key is None:
                continue
            key = str(key)
            if key in chunk_index:
                ctx.append(chunk_index[key])
                break
        if len(ctx) >= max_contexts:
            break
    return ctx


# -----------------------------
# Core eval loop
# -----------------------------

def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(p * (len(xs_sorted) - 1))
    return xs_sorted[idx]


def run_eval(
    eval_path: Path,
    out_csv: Path,
    alpha: float,
    top_k: int,
    use_local_lora: Optional[bool] = None,
) -> None:
    data = load_eval_dataset(eval_path)
    rows: List[Dict[str, Any]] = []

    n = len(data)
    print(f"Running eval on {n} questions from {eval_path}...")
    print(f"alpha={alpha} | top_k={top_k}")

    # Allow eval harness to force local generation on/off (optional)
    if use_local_lora is not None:
        os.environ["USE_LOCAL_LORA"] = "1" if use_local_lora else "0"
        print(f"USE_LOCAL_LORA forced to {os.environ['USE_LOCAL_LORA']}")

    exact_scores: List[float] = []
    semantic_scores: List[float] = []
    grounding_scores: List[float] = []
    latencies: List[float] = []

    # For optional RAGAS
    ragas_questions: List[str] = []
    ragas_answers: List[str] = []
    ragas_contexts: List[List[str]] = []

    chunk_index = load_chunk_index(CHUNKS_PATH)

    for item in data:
        qid = item["id"]
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_keywords = item.get("expected_keywords", []) or []
        qtype = item.get("type", "") or "unknown"

        print(f"\n→ [{qid}] ({qtype}) {question}")

        t0 = time.time()
        result = pipeline.answer_query(question, top_k=top_k, alpha=alpha)
        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)

        answer = (result.get("answer", "") or "").strip()
        citations = result.get("citations", []) or []

        em = exact_match(answer, expected_answer)
        sem = keyword_f1(answer, expected_keywords)
        grd = grounding_score(answer, citations, expected_keywords)

        # Reward safe refusals for edge/safety questions even if no citations
        if qtype in EDGE_TYPES and is_safe_refusal(answer):
            sem = 1.0
            grd = 1.0
            # exact match is not meaningful here; leave as computed

        exact_scores.append(em)
        semantic_scores.append(sem)
        grounding_scores.append(grd)

        print(f"  latency: {dt_ms:.1f} ms")
        print(f"  exact_match: {em:.2f}, semantic: {sem:.2f}, grounding: {grd:.2f}")

        # RAGAS contexts
        ctx_texts = resolve_contexts_from_citations(citations, chunk_index, max_contexts=6)
        ragas_questions.append(question)
        ragas_answers.append(answer)
        ragas_contexts.append(ctx_texts)

        rows.append(
            {
                "id": qid,
                "type": qtype,
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "expected_keywords": "|".join([str(x) for x in expected_keywords]),
                "exact_match": float(em),
                "semantic_keyword_f1": float(sem),
                "grounding_score": float(grd),
                "latency_ms": float(dt_ms),
                "citations": citations,
            }
        )

    # -----------------------------
    # Win/Loss rule (match your printed summary)
    # -----------------------------
    # You were printing: "Win rate (semantic F1 + grounding >= 0.5)"
    # So we implement: (sem + grd) >= 0.5
    def is_win(row: Dict[str, Any]) -> bool:
        sem = float(row["semantic_keyword_f1"])
        grd = float(row["grounding_score"])
        return (sem + grd) >= 0.5

    for row in rows:
        row["win"] = 1 if is_win(row) else 0

    # -----------------------------
    # Aggregate summary
    # -----------------------------
    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    exact_avg = _avg(exact_scores)
    sem_avg = _avg(semantic_scores)
    grd_avg = _avg(grounding_scores)
    lat_avg = _avg(latencies)
    lat_p50 = percentile(latencies, 0.50)
    lat_p95 = percentile(latencies, 0.95)

    total_wins = sum(int(r["win"]) for r in rows)
    win_rate = total_wins / n if n else 0.0

    print("\n================ EVAL SUMMARY ================")
    print(f"Total questions: {n}")
    print(f"Win rate (semantic F1 + grounding >= 0.5): {total_wins} / {n} = {win_rate:.1%}")
    print(f"Exact match avg:         {exact_avg:.3f}")
    print(f"Semantic keyword F1 avg: {sem_avg:.3f}")
    print(f"Grounding score avg:     {grd_avg:.3f}")
    print(f"Latency avg: {lat_avg:.1f} ms | p50: {lat_p50:.1f} ms | p95: {lat_p95:.1f} ms")

    # Breakdown by type
    by_type: Dict[str, Dict[str, int]] = {}
    for row in rows:
        t = row.get("type", "unknown") or "unknown"
        by_type.setdefault(t, {"wins": 0, "total": 0})
        by_type[t]["total"] += 1
        by_type[t]["wins"] += int(row["win"])

    print("\nBreakdown by type:")
    for t, stats in by_type.items():
        tot = stats["total"]
        wins = stats["wins"]
        rate = wins / tot if tot else 0.0
        print(f"  {t:12s}: {wins:3d} / {tot:3d} = {rate:.1%}")

    # -----------------------------
    # Write CSV
    # -----------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
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

    print(f"\n📊 Detailed report written to: {out_csv}")

    # Optional RAGAS
    maybe_run_ragas(ragas_questions, ragas_answers, ragas_contexts)


# -----------------------------
# Optional RAGAS hook
# -----------------------------

def maybe_run_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
) -> None:
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

    # `result` is dict-like
    for metric_name, value in result.items():
        if isinstance(value, list) and value:
            avg = sum(value) / len(value)
            print(f"{metric_name}: {avg:.3f}")
        else:
            print(f"{metric_name}: {value}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=str, default=str(DEFAULT_EVAL_PATH), help="Path to eval JSON (baseline.json)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV), help="Path to output CSV (report.csv)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Hybrid retriever alpha")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retriever top_k")
    p.add_argument(
        "--use-local-lora",
        choices=["0", "1"],
        default=None,
        help="Force local LoRA generation on/off for this run (overrides env)",
    )
    return p.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--use-local-lora", type=int, default=0)
    parser.add_argument(
        "--eval-path",
        type=str,
        default=str(Path(__file__).parent / "baseline.json"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path",
    )

    args = parser.parse_args()

    # Toggle local LoRA via env var (pipeline reads this)
    if args.use_local_lora:
        os.environ["USE_LOCAL_LORA"] = "1"

    eval_path = Path(args.eval_path)

    if args.out:
        out_csv = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "lora" if args.use_local_lora else "openai"
        out_csv = (
            Path(__file__).parent
            / f"report_{mode}_a{args.alpha}_k{args.top_k}_{stamp}.csv"
        )

    run_eval(
        eval_path=eval_path,
        out_csv=out_csv,
        alpha=args.alpha,
        top_k=args.top_k,
    )