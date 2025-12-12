#!/usr/bin/env python
"""
Build finetune datasets from evals:

Inputs:
  - evals/baseline.json
  - evals/report.csv

Outputs:
  - data/finetune/from_evals.jsonl
      -> uses baseline.expected_answer as output
  - data/finetune/from_eval_wins.jsonl
      -> uses model answers from report.csv where
         semantic_keyword_f1 + grounding_score >= 0.5
  - data/finetune/combined_finetune.jsonl
      -> concatenation of both (with simple de-duplication)

All records follow Alpaca-style schema:
{
  "instruction": "...",
  "input": "QUESTION:\\n...",
  "output": "..."
}
"""

import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "evals" / "baseline.json"
REPORT_PATH = ROOT / "evals" / "report.csv"

FINETUNE_DIR = ROOT / "data" / "finetune"
FROM_EVALS_PATH = FINETUNE_DIR / "from_evals.jsonl"
FROM_WINS_PATH = FINETUNE_DIR / "from_eval_wins.jsonl"
COMBINED_PATH = FINETUNE_DIR / "combined_finetune.jsonl"

FINETUNE_DIR.mkdir(parents=True, exist_ok=True)


def _make_instruction(item_type: str) -> str:
    """
    Map eval 'type' into a high-level instruction string.
    """
    item_type = (item_type or "").strip()
    if item_type == "edge":
        return (
            "Refuse safely if the question is unsafe or clearly out of scope for "
            "the MITRE ATT&CK and AI security documents; otherwise give a brief "
            "answer based only on those documents."
        )
    return (
        "Answer the security question based only on what is covered in the "
        "MITRE ATT&CK and AI security documents. Be concise and factual."
    )


def load_baseline() -> List[Dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval dataset not found at {EVAL_PATH}")
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_index() -> Dict[str, Dict]:
    data = load_baseline()
    return {item["id"]: item for item in data}


def build_from_evals() -> List[Dict]:
    """
    Use baseline.expected_answer as the ideal output.
    """
    data = load_baseline()
    records: List[Dict] = []

    for item in data:
        question = (item.get("question") or "").strip()
        expected_answer = (item.get("expected_answer") or "").strip()
        item_type = item.get("type") or item.get("qtype") or ""

        if not question or not expected_answer:
            continue

        records.append(
            {
                "instruction": _make_instruction(item_type),
                "input": f"QUESTION:\n{question}",
                "output": expected_answer,
            }
        )

    print(f"[from_evals] built {len(records)} records")
    return records


def build_from_wins() -> List[Dict]:
    """
    Use model answers from report.csv where sem + grd >= 0.5.
    """
    if not REPORT_PATH.exists():
        print(f"[from_eval_wins] report.csv not found at {REPORT_PATH}, skipping.")
        return []

    baseline_by_id = load_baseline_index()
    records: List[Dict] = []

    with REPORT_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sem = float(row.get("semantic_keyword_f1", 0.0))
                grd = float(row.get("grounding_score", 0.0))
            except ValueError:
                continue

            if sem + grd < 0.5:
                continue  # not a win

            qid = row.get("id")
            answer = (row.get("answer") or "").strip()
            if not qid or not answer:
                continue

            base = baseline_by_id.get(qid)
            if not base:
                continue

            question = (base.get("question") or "").strip()
            item_type = base.get("type") or base.get("qtype") or ""

            if not question:
                continue

            records.append(
                {
                    "instruction": _make_instruction(item_type),
                    "input": f"QUESTION:\n{question}",
                    "output": answer,
                }
            )

    print(f"[from_eval_wins] built {len(records)} records")
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records → {path}")


def main() -> None:
    # 1) Build both datasets in memory
    from_evals = build_from_evals()
    from_wins = build_from_wins()

    # 2) Write individual files
    write_jsonl(FROM_EVALS_PATH, from_evals)
    if from_wins:
        write_jsonl(FROM_WINS_PATH, from_wins)

    # 3) Combined file with simple de-duplication
    #    We de-dup based on (instruction, input, output) triple.
    seen = set()
    combined: List[Dict] = []

    for rec in from_evals + from_wins:
        key = (rec["instruction"], rec["input"], rec["output"])
        if key in seen:
            continue
        seen.add(key)
        combined.append(rec)

    write_jsonl(COMBINED_PATH, combined)


if __name__ == "__main__":
    main()