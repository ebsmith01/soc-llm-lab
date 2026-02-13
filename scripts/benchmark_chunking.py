#!/usr/bin/env python3
"""
Quick benchmark for Day 2.
Compares a few chunk sizes and prints chunk counts + average size.

Usage:
  python scripts/benchmark_chunking.py
"""

from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chunks import chunk_text, count_tokens

CLEAN_TXT_DIR = PROJECT_ROOT / "data" / "clean_txt"


def main():
    txt_files = sorted(CLEAN_TXT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {CLEAN_TXT_DIR}")

    sizes = [500]
    overlap = 75

    for sz in sizes:
        all_chunks = []
        for p in txt_files:
            text = p.read_text(encoding="utf-8")
            all_chunks.extend(chunk_text(text, chunk_size_tokens=sz, chunk_overlap_tokens=overlap))

        token_lens = [count_tokens(c) for c in all_chunks]
        print(
            f"chunk_size={sz} overlap={overlap} "
            f"→ chunks={len(all_chunks)} avg_tokens={int(mean(token_lens))}"
        )


if __name__ == "__main__":
    main()
