# evals/run_eval.py
"""
Evaluation Harness (NO RAGAS)

This script:
  1) Loads an eval dataset (baseline.json)
  2) Calls pipeline.answer_query(question)
  3) Computes:
       - exact_match (strict string)
       - semantic_keyword_f1 (keyword recall-ish)
       - grounding_score (citations present + keyword overlap)
  4) Writes a CSV report
  5) Prints aggregate metrics + breakdown by type

NEW (RERANK SUPPORT)
-------------------
Adds CLI flags to control retrieval candidate pool + reranking, and forwards them
into pipeline.answer_query().

Flags:
  --semantic-top-k        (candidate pool size per retriever)
  --use-reranker          (0/1)
  --reranker-model        (HF model id)
  --rerank-top-n          (how many fused candidates to rerank)
  --rerank-lambda         (blend weight for rerank vs hybrid)
  --rerank-max-chars      (snippet size for reranker input, if your retriever uses it)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

DEFAULT_ALPHA = 0.6
DEFAULT_TOP_K = 4

# NEW: retrieval + rerank defaults (match your tuned baseline + rerank experiments)
DEFAULT_SEMANTIC_TOP_K = 60
DEFAULT_USE_RERANKER = 0
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_TOP_N = 60
DEFAULT_RERANK_LAMBDA = 0.70
DEFAULT_RERANK_MAX_CHARS = 1800

EDGE_TYPES = {"edge", "out_of_scope", "safety"}

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


# -----------------------------
# Helpers for text + metrics
# -----------------------------
def _normalize(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def exact_match(pred: str, gold: str) -> float:
    gold = gold or ""
    return 1.0 if gold and _normalize(pred) == _normalize(gold) else 0.0


def keyword_f1(pred: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0

    pred_norm = _normalize(pred)
    gold = [kw.lower().strip() for kw in expected_keywords if kw and str(kw).strip()]
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    hits = sum(1 for kw in gold_set if kw in pred_norm)
    return hits / len(gold_set)


def grounding_score(answer: str, citations: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    if not citations:
        return 0.0
    return keyword_f1(answer, expected_keywords)


def is_safe_refusal(answer: str) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(p in lower for p in SAFE_REFUSAL_PHRASES)


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = int(p * (len(xs_sorted) - 1))
    return xs_sorted[idx]


# -----------------------------
# Dataset loading
# -----------------------------
def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Core eval
# -----------------------------
def run_eval(
    eval_path: Path,
    out_csv: Path,
    alpha: float,
    top_k: int,
    semantic_top_k: int,
    use_reranker: bool,
    reranker_model: str,
    rerank_top_n: int,
    rerank_lambda: float,
    rerank_max_chars: int,
    use_local_lora: Optional[bool] = None,
) -> None:
    data = load_eval_dataset(eval_path)
    rows: List[Dict[str, Any]] = []

    n = len(data)
    print(f"Running eval on {n} questions from {eval_path}...")
    print(
        "alpha={a} | top_k={k} | semantic_top_k={stk} | use_reranker={ur} | "
        "rerank_top_n={rtn} | rerank_lambda={rl:.2f}".format(
            a=alpha, k=top_k, stk=semantic_top_k, ur=int(use_reranker), rtn=rerank_top_n, rl=rerank_lambda
        )
    )
    if use_reranker:
        print(f"reranker_model={reranker_model}")

    if use_local_lora is not None:
        os.environ["USE_LOCAL_LORA"] = "1" if use_local_lora else "0"
        pipeline.USE_LOCAL_LORA = bool(use_local_lora)
        print(f"USE_LOCAL_LORA forced to {os.environ['USE_LOCAL_LORA']}")

    exact_scores: List[float] = []
    semantic_scores: List[float] = []
    grounding_scores: List[float] = []
    latencies: List[float] = []

    for item in data:
        qid = item.get("id", "")
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "") or ""
        expected_keywords = item.get("expected_keywords", []) or []
        qtype = (item.get("type", "") or "unknown").strip()

        print(f"\n→ [{qid}] ({qtype}) {question}")

        t0 = time.time()

        # Forward all knobs to pipeline (pipeline.answer_query should accept these kwargs)
        result = pipeline.answer_query(
            question,
            top_k=top_k,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_n=rerank_top_n,
            rerank_lambda=rerank_lambda,
            rerank_max_chars=rerank_max_chars,
        )

        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)

        answer = (result.get("answer", "") or "").strip()
        citations = result.get("citations", []) or []

        em = exact_match(answer, expected_answer)
        sem = keyword_f1(answer, expected_keywords)
        grd = grounding_score(answer, citations, expected_keywords)

        if qtype in EDGE_TYPES and is_safe_refusal(answer):
            sem = 1.0
            grd = 1.0

        exact_scores.append(em)
        semantic_scores.append(sem)
        grounding_scores.append(grd)

        print(f"  latency: {dt_ms:.1f} ms")
        print(f"  exact_match: {em:.2f}, semantic: {sem:.2f}, grounding: {grd:.2f}")

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

    def is_win(row: Dict[str, Any]) -> bool:
        return float(row["semantic_keyword_f1"]) >= 0.5 and float(row["grounding_score"]) >= 0.5

    for row in rows:
        row["win"] = 1 if is_win(row) else 0

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


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG evaluation (no RAGAS)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    # NEW: candidate pool + rerank knobs
    p.add_argument("--semantic-top-k", type=int, default=DEFAULT_SEMANTIC_TOP_K)
    p.add_argument("--use-reranker", type=int, choices=[0, 1], default=DEFAULT_USE_RERANKER)
    p.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL)
    p.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    p.add_argument("--rerank-lambda", type=float, default=DEFAULT_RERANK_LAMBDA)
    p.add_argument("--rerank-max-chars", type=int, default=DEFAULT_RERANK_MAX_CHARS)

    p.add_argument("--use-local-lora", type=int, choices=[0, 1], default=0)
    p.add_argument("--eval-path", type=str, default=str(DEFAULT_EVAL_PATH))
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path. If omitted, we auto-name with timestamp.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    eval_path = Path(args.eval_path)

    if args.out:
        out_csv = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "lora" if args.use_local_lora else "openai"
        out_csv = DEFAULT_OUT_DIR / f"report_{mode}_a{args.alpha}_k{args.top_k}_{stamp}.csv"

    run_eval(
        eval_path=eval_path,
        out_csv=out_csv,
        alpha=float(args.alpha),
        top_k=int(args.top_k),
        semantic_top_k=int(args.semantic_top_k),
        use_reranker=bool(args.use_reranker),
        reranker_model=str(args.reranker_model),
        rerank_top_n=int(args.rerank_top_n),
        rerank_lambda=float(args.rerank_lambda),
        rerank_max_chars=int(args.rerank_max_chars),
        use_local_lora=bool(args.use_local_lora),
    )