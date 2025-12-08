
from dotenv import load_dotenv


# load_dotenv is pulled in so a .env file can populate environment variables before first access.
load_dotenv()  # ensure .env values (OPENAI_API_KEY, etc.) are available
"""
Build BM25 + embeddings (sentence-transformers) from chunks.jsonl.

Outputs (in data/processed):
  - bm25.pkl
  - tokenized_corpus.json
  - embeddings.npy
"""

from pathlib import Path
import json
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_PATH = PROC_DIR / "chunks.jsonl"
BM25_PATH = PROC_DIR / "bm25.pkl"
TOK_PATH = PROC_DIR / "tokenized_corpus.json"
EMB_PATH = PROC_DIR / "embeddings.npy"


def build_indexes():
    print(f"📄 Loading chunks from {CHUNKS_PATH} ...")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {CHUNKS_PATH}")

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]

    # -----------------------------
    # BM25
    # -----------------------------
    print("📚 Building BM25 index ...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25, f)

    TOK_PATH.write_text(json.dumps(tokenized), encoding="utf-8")

    # -----------------------------
    # SentenceTransformer embeddings
    # -----------------------------
    print("🔢 Embedding chunks with sentence-transformers ...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = st_model.encode(texts, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    np.save(EMB_PATH, vectors)

    print("\n✅ Index build complete!")
    print(f"  - {BM25_PATH}")
    print(f"  - {TOK_PATH}")
    print(f"  - {EMB_PATH}")


if __name__ == "__main__":
    build_indexes()