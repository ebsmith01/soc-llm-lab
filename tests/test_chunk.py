"""
Day 2 unit tests for chunking.

Run:
  pytest -q
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def test_chunks_file_exists():
    assert OUT_PATH.exists(), "chunks.jsonl not found. Run scripts/chunk.py"


def test_chunks_not_empty_and_reasonable_count():
    lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 5, "Too few chunks; check ingestion/clean_txt inputs."

    for line in lines:
        rec = json.loads(line)
        assert rec["text"].strip() != "", "Found empty chunk text."
        assert "id" in rec and rec["id"], "Missing id."
        assert "source" in rec and rec["source"], "Missing source."