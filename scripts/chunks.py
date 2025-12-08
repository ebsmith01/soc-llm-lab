#!/usr/bin/env python3
"""
Day 2 — Chunking Strategy + Implementation

Reads cleaned .txt files from data/clean_txt/
Splits into ~token-sized chunks with overlap
Writes JSONL chunks to data/processed/chunks.jsonl

Each JSONL line:
{
  "id": "...",
  "text": "...",
  "source": "filename.txt",
  "page_num": null
}

Usage:
  python scripts/chunk.py
"""

from pathlib import Path
import json
import re
from typing import List, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_TXT_DIR = PROJECT_ROOT / "data" / "clean_txt"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# -----------------------------
# Token counting (best-effort)
# -----------------------------
def count_tokens(text: str) -> int:
    """
    Tries to use tiktoken if installed (closest to real LLM tokens).
    Falls back to rough estimate if not.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # rough fallback: ~4 chars/token average in English
        return max(1, len(text) // 4)


# -----------------------------
# Very light sentence awareness
# -----------------------------
_SENT_END = re.compile(r"(?<=[.!?])\s+")

def sentence_aware_separators() -> List[str]:
    """
    RecursiveCharacterTextSplitter uses a list of separators.
    Putting sentence boundaries early makes splits more sentence-friendly.
    """
    return [
        "\n\n",          # paragraphs
        "\n",            # line breaks
        ". ", "? ", "! ",# sentence-ish
        " ",             # words
        ""               # fallback to characters
    ]


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(
    text: str,
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 80
) -> List[str]:
    """
    Chunk a text blob into ~token chunks with overlap.
    We approximate tokens using a custom length function.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=sentence_aware_separators(),
        length_function=count_tokens,  # makes chunk_size "token-like"
    )
    return splitter.split_text(text)


def build_chunks_for_file(path: Path, chunk_size_tokens=450, overlap_tokens=100) -> List[Dict]:
    """
    Read one clean txt file and return list of chunk dicts.
    """
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, chunk_size_tokens, overlap_tokens)

    out = []
    for i, c in enumerate(chunks):
        c = c.strip()
        if not c:
            continue
        out.append({
            "id": f"{path.stem}-{i}",
            "text": c,
            "source": path.name,
            "page_num": None,  # page Num unknown at txt level
        })
    return out


def main():
    if not CLEAN_TXT_DIR.exists():
        raise FileNotFoundError(
            f"{CLEAN_TXT_DIR} does not exist. Day 1 should create data/clean_txt/*.txt"
        )

    txt_files = sorted(CLEAN_TXT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {CLEAN_TXT_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for txt in txt_files:
            print(f"Chunking {txt.name}...")
            records = build_chunks_for_file(txt)
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  → {len(records)} chunks")
            total += len(records)

    print(f"\nDone. Wrote {total} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()