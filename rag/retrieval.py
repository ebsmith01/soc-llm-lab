"""
HIGH-LEVEL IDEA
---------------
We do two searches:

1) BM25 lexical search:
   - exact keyword matching
   - strong for technical terms, IDs, proper nouns

2) Semantic search:
   - embeddings + dot-product similarity
   - strong for paraphrases / conceptual queries

Then we:
  a) collect top candidates from both
  b) normalize their scores to same scale
  c) fuse with weighted average:
     hybrid = alpha * bm25 + (1-alpha) * semantic
  d) return top-k chunks

OUTPUT FORMAT
-------------
Each hit is a dict:
{
  "id": ...,
  "text": ...,
  "score": hybrid_score,
  "bm25_score": raw bm25 score,
  "semantic_score": raw dot-product score,
  "source": ...,
  "page_num": ...,
  "metadata": {...}
}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import json
import pickle
import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths for loading indexes
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
TOKENIZED_PATH = PROCESSED_DIR / "tokenized_corpus.json"
EMB_PATH = PROCESSED_DIR / "embeddings.npy"


# -----------------------------
# Tokenization for BM25 queries
# -----------------------------

def tokenize(text: str) -> List[str]:
    """
    Must match the tokenizer used to build bm25.pkl.

    Baseline: lowercase whitespace tokens.
    """
    return text.lower().split()


# -----------------------------
# Score normalization
# -----------------------------

def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0,1].

    BM25 scores and semantic similarities live on different scales.
    Normalizing inside the candidate pool makes them comparable.

    If all scores are equal:
      we return zeros to avoid divide-by-zero.
    """
    if scores.size == 0:
        return scores

    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)

    return (scores - mn) / (mx - mn)


# -----------------------------
# Hybrid Retriever
# -----------------------------

class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25: BM25Okapi,
        tokenized_corpus: List[List[str]],
        embeddings: np.ndarray,
        alpha: float = 0.7,
    ):
        """
        alpha controls hybrid weighting:
          hybrid_score = alpha * bm25_score + (1 - alpha) * semantic_score
        """
        self.chunks = chunks
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.embeddings = embeddings
        self.alpha = alpha

        # Query embedder (lazy init)
        self._embedder: Optional[SentenceTransformer] = None
        self._logger = logging.getLogger(__name__)

    # -----------------------------
    # Loading helpers
    # -----------------------------

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict[str, Any]]:
        """
        Reads JSONL chunks into memory in original corpus order.
        That order must align with BM25 + embeddings row indexes.
        """
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run scripts/ingest.py (Day 2).")
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: Optional[Path] = None,
        alpha: float = 0.8,
    ) -> "HybridRetriever":
        """
        Load chunks + BM25 + embeddings from data/processed.

        Args:
            processed_dir: directory where processed artifacts are stored.
                           Defaults to PROJECT_ROOT/data/processed.
            alpha: hybrid weighting between BM25 and semantic similarity.
        """
        if processed_dir is None:
            processed_dir = PROCESSED_DIR

        chunks_path = processed_dir / "chunks.jsonl"
        bm25_path = processed_dir / "bm25.pkl"
        tokenized_path = processed_dir / "tokenized_corpus.json"
        embeddings_path = processed_dir / "embeddings.npy"

        # 1) Load chunks from JSONL
        chunks = cls._load_chunks(chunks_path)

        # 2) Load BM25 + tokenized corpus
        if not bm25_path.exists():
            raise FileNotFoundError(f"{bm25_path} missing. Run scripts/build_indexes.py.")
        with bm25_path.open("rb") as f:
            bm25 = pickle.load(f)

        if not tokenized_path.exists():
            raise FileNotFoundError(f"{tokenized_path} missing. Run scripts/build_indexes.py.")
        with tokenized_path.open("r", encoding="utf-8") as f:
            tokenized_corpus = json.load(f)

        # 3) Load embeddings matrix
        if not embeddings_path.exists():
            raise FileNotFoundError(f"{embeddings_path} missing. Run scripts/build_indexes.py.")
        embeddings = np.load(embeddings_path)

        return cls(
            chunks=chunks,
            bm25=bm25,
            tokenized_corpus=tokenized_corpus,
            embeddings=embeddings,
            alpha=alpha,
        )
    
    from rag.config import current_config



    # -----------------------------
    # Query embedding
    # -----------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Produces an embedding vector for the query.

        Returns:
            np.ndarray of shape (1, dim)
        """
        if self._embedder is None:
            # Use the SAME model as build_indexes.py
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(PROJECT_ROOT / ".cache" / "models"),
            )

        vec = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # encode([...]) returns shape (1, dim)
        q_vec = np.asarray(vec, dtype=np.float32)
        return q_vec  # shape: (1, dim)

    # -----------------------------
    # BM25 search
    # -----------------------------

    def _bm25_search(self, query: str) -> np.ndarray:
        """
        Returns BM25 scores for *all* chunks.
        Shape: (n_chunks,)
        """
        q_tokens = tokenize(query)
        return np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

    # -----------------------------
    # Semantic search (brute-force, no FAISS)
    # -----------------------------

    def _semantic_search(self, query: str, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns top_k semantic hits using brute-force dot-product.

        For small corpora (like this lab), brute-force over all embeddings
        is totally fine and avoids complexity/bugs from FAISS integration.

        Returns:
          (indices, scores)
        """
        q_vec = self._embed_query(query)  # shape: (1, dim)
        # embeddings: (n_chunks, dim)
        # q_vec[0]: (dim,)
        sims = self.embeddings @ q_vec[0]  # dot-product similarity
        idxs = np.argsort(-sims)[:top_k]
        return idxs, sims[idxs]

    # -----------------------------
    # Public hybrid search
    # -----------------------------

    def search(self, query: str, k: int = 5, semantic_top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval in 4 stages:

        1) BM25 scores for all chunks
        2) semantic top_k by embeddings (brute-force)
        3) build candidate set = union(bm25_top, semantic_top)
        4) normalize and fuse scores, return global top-k

        semantic_top_k:
          candidate pool size per retriever.
          We gather ~20 from each to fuse.
        """

        # 1) Lexical scores for all chunks
        bm25_scores = self._bm25_search(query)

        # 2) Semantic top hits
        sem_idxs, sem_scores = self._semantic_search(query, top_k=semantic_top_k)

        # Candidate BM25 top hits
        bm25_top = np.argsort(-bm25_scores)[:semantic_top_k]

        # 3) Candidate union
        cand_idxs = np.unique(np.concatenate([bm25_top, sem_idxs]))

        # Pull candidate BM25 scores
        cand_bm25 = bm25_scores[cand_idxs]

        # Start semantic scores as zeros; fill if in sem_idxs
        cand_sem = np.zeros_like(cand_bm25)
        sem_map = {int(i): float(s) for i, s in zip(sem_idxs, sem_scores)}
        for j, idx in enumerate(cand_idxs):
            cand_sem[j] = sem_map.get(int(idx), 0.0)

        # 4) Normalize each score type in candidate pool
        nb = minmax_normalize(cand_bm25)
        ns = minmax_normalize(cand_sem)

        # Weighted hybrid fusion
        hybrid = self.alpha * nb + (1.0 - self.alpha) * ns

        # Choose top-k by hybrid score
        order = np.argsort(-hybrid)[:k]

        results: List[Dict[str, Any]] = []
        for rank_pos in order:
            i = int(cand_idxs[rank_pos])
            c = self.chunks[i]

            results.append(
                {
                    "id": c.get("id", f"chunk-{i}"),
                    "text": c["text"],
                    "score": float(hybrid[rank_pos]),
                    "bm25_score": float(cand_bm25[rank_pos]),
                    "semantic_score": float(cand_sem[rank_pos]),
                    "source": c.get("source"),
                    "page_num": c.get("page_num"),
                    "metadata": c.get("metadata", {}),
                }
            )

        return results
    