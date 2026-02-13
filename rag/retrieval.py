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
  d) optionally rerank the fused candidate pool with a cross-encoder
  e) return top-k chunks

RERANKING
---------
Cross-encoder reranking scores (query, passage) jointly to improve ranking precision.
To prevent the reranker from overriding a good hybrid baseline, we BLEND rerank + hybrid:

  blended = λ * zscore(rerank) + (1-λ) * zscore(hybrid)

OUTPUT FORMAT
-------------
Each hit is a dict:
{
  "id": ...,
  "text": ...,
  "score": hybrid_score,        # fused score (pre-rerank)
  "final_score": final_score,   # blended rerank+hybrid (only if reranking enabled)
  "bm25_score": raw bm25 score,
  "semantic_score": raw dot-product score,
  "rerank_score": cross-encoder score (only if reranking enabled),
  "source": ...,
  "page_num": ...,
  "metadata": {...}
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# -----------------------------
# Paths for loading indexes
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
TOKENIZED_PATH = PROCESSED_DIR / "tokenized_corpus.json"
EMB_PATH = PROCESSED_DIR / "embeddings.npy"


def tokenize(text: str) -> List[str]:
    """Must match the tokenizer used to build bm25.pkl (lowercase whitespace tokens)."""
    return text.lower().split()


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0,1]."""
    if scores.size == 0:
        return scores
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def zscore_normalize(scores: np.ndarray) -> np.ndarray:
    """Z-score normalization (helps blend rerank + hybrid on comparable scale)."""
    if scores.size == 0:
        return scores
    mu = float(scores.mean())
    sd = float(scores.std())
    if sd == 0.0:
        return np.zeros_like(scores)
    return (scores - mu) / sd


def snippet_head_tail(text: str, max_chars: int = 1800) -> str:
    """
    Better than head-only truncation: keeps evidence that may appear near the end.
    Cross-encoders will truncate internally (256/512 tokens), so we choose a stable snippet.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25: BM25Okapi,
        tokenized_corpus: List[List[str]],
        embeddings: np.ndarray,
        # IMPORTANT: default to your tuned best
        alpha: float = 0.60,
        # ---- Reranking knobs ----
        use_reranker: bool = False,
        # IMPORTANT: default to a strong domain-robust reranker
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        # IMPORTANT: reranker must see enough candidates to help
        rerank_top_n: int = 60,
        rerank_lambda: float = 0.70,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.embeddings = embeddings
        self.alpha = float(alpha)

        self.use_reranker = bool(use_reranker)
        self.reranker_model = reranker_model
        self.rerank_top_n = int(rerank_top_n)
        self.rerank_lambda = float(rerank_lambda)

        self._embedder: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        self._logger = logging.getLogger(__name__)

    # -----------------------------
    # Loading helpers
    # -----------------------------

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict[str, Any]]:
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
        alpha: float = 0.60,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_top_n: int = 60,
        rerank_lambda: float = 0.70,
    ) -> "HybridRetriever":
        if processed_dir is None:
            processed_dir = PROCESSED_DIR

        chunks = cls._load_chunks(processed_dir / "chunks.jsonl")

        bm25_path = processed_dir / "bm25.pkl"
        tokenized_path = processed_dir / "tokenized_corpus.json"
        embeddings_path = processed_dir / "embeddings.npy"

        if not bm25_path.exists():
            raise FileNotFoundError(f"{bm25_path} missing. Run scripts/build_indexes.py.")
        with bm25_path.open("rb") as f:
            bm25 = pickle.load(f)

        if not tokenized_path.exists():
            raise FileNotFoundError(f"{tokenized_path} missing. Run scripts/build_indexes.py.")
        with tokenized_path.open("r", encoding="utf-8") as f:
            tokenized_corpus = json.load(f)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"{embeddings_path} missing. Run scripts/build_indexes.py.")
        embeddings = np.load(embeddings_path)

        return cls(
            chunks=chunks,
            bm25=bm25,
            tokenized_corpus=tokenized_corpus,
            embeddings=embeddings,
            alpha=alpha,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            rerank_top_n=rerank_top_n,
            rerank_lambda=rerank_lambda,
        )

    # -----------------------------
    # Query embedding
    # -----------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=str(PROJECT_ROOT / ".cache" / "models"),
            )
        vec = self._embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)

    # -----------------------------
    # Reranker
    # -----------------------------

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(
                self.reranker_model,
                cache_folder=str(PROJECT_ROOT / ".cache" / "models"),
            )
        return self._reranker

    def _rerank_scores(self, query: str, idxs: np.ndarray) -> np.ndarray:
        reranker = self._get_reranker()
        texts = [snippet_head_tail(self.chunks[int(i)]["text"]) for i in idxs]
        pairs = [(query, t) for t in texts]
        scores = reranker.predict(pairs)
        return np.asarray(scores, dtype=np.float32)

    # -----------------------------
    # BM25 search
    # -----------------------------

    def _bm25_search(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        return np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

    # -----------------------------
    # Semantic search (brute-force)
    # -----------------------------

    def _semantic_search(self, query: str, top_k: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        q_vec = self._embed_query(query)          # (1, dim)
        sims = self.embeddings @ q_vec[0]         # (n_chunks,)
        idxs = np.argsort(-sims)[:top_k]
        return idxs, sims[idxs]

    # -----------------------------
    # Public hybrid search
    # -----------------------------

    def search(
        self,
        query: str,
        # IMPORTANT: your tuned winner
        k: int = 4,
        # IMPORTANT: enlarge pool so reranker has something to improve
        semantic_top_k: int = 60,
        rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        do_rerank = self.use_reranker if rerank is None else bool(rerank)

        bm25_scores = self._bm25_search(query)
        sem_idxs, sem_scores = self._semantic_search(query, top_k=semantic_top_k)
        bm25_top = np.argsort(-bm25_scores)[:semantic_top_k]

        cand_idxs = np.unique(np.concatenate([bm25_top, sem_idxs]))
        cand_bm25 = bm25_scores[cand_idxs]

        cand_sem = np.zeros_like(cand_bm25)
        sem_map = {int(i): float(s) for i, s in zip(sem_idxs, sem_scores)}
        for j, idx in enumerate(cand_idxs):
            cand_sem[j] = sem_map.get(int(idx), 0.0)

        nb = minmax_normalize(cand_bm25)
        ns = minmax_normalize(cand_sem)
        hybrid = self.alpha * nb + (1.0 - self.alpha) * ns

        fused_order = np.argsort(-hybrid)
        cand_idxs = cand_idxs[fused_order]
        cand_bm25 = cand_bm25[fused_order]
        cand_sem = cand_sem[fused_order]
        hybrid = hybrid[fused_order]

        final_order = cand_idxs
        rerank_scores: Optional[np.ndarray] = None
        blended_scores: Optional[np.ndarray] = None

        if do_rerank:
            top_n = min(self.rerank_top_n, cand_idxs.size)
            top_idxs = cand_idxs[:top_n]

            rerank_scores = self._rerank_scores(query, top_idxs)

            lam = self.rerank_lambda
            r_norm = zscore_normalize(rerank_scores)
            h_norm = zscore_normalize(hybrid[:top_n])
            blended = lam * r_norm + (1.0 - lam) * h_norm

            top_order = np.argsort(-blended)
            final_order = np.concatenate([top_idxs[top_order], cand_idxs[top_n:]])
            blended_scores = blended

        results: List[Dict[str, Any]] = []
        for idx in final_order[:k]:
            i = int(idx)
            c = self.chunks[i]

            fused_pos = int(np.where(cand_idxs == i)[0][0])

            row: Dict[str, Any] = {
                "id": c.get("id", f"chunk-{i}"),
                "text": c["text"],
                "score": float(hybrid[fused_pos]),
                "bm25_score": float(cand_bm25[fused_pos]),
                "semantic_score": float(cand_sem[fused_pos]),
                "source": c.get("source"),
                "page_num": c.get("page_num"),
                "metadata": c.get("metadata", {}),
            }

            if do_rerank and rerank_scores is not None and blended_scores is not None:
                top_n = min(self.rerank_top_n, cand_idxs.size)
                top_idxs = cand_idxs[:top_n]
                map_r = {int(ti): float(rs) for ti, rs in zip(top_idxs, rerank_scores)}
                map_f = {int(ti): float(fs) for ti, fs in zip(top_idxs, blended_scores)}
                if i in map_r:
                    row["rerank_score"] = map_r[i]
                    row["final_score"] = map_f[i]

            results.append(row)

        return results