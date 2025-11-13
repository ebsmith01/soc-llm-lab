from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json, os
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

_DEFAULT_CHUNKS = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.jsonl"
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(_DEFAULT_CHUNKS)))

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]

class HybridRetriever:
    def __init__(self, embed_model: str = "intfloat/e5-base-v2"):
        self.chunks: List[Chunk] = []
        self._bm25 = None
        self._embedder = None
        self._dense_matrix = None

        self._load_corpus()
        self._build_bm25()
        self._build_dense(embed_model)

    def _load_corpus(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.chunks.append(Chunk(id=obj["id"], text=obj["text"], meta=obj.get("meta", {})))

    def _build_bm25(self):
        tokenized = [c.text.split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _build_dense(self, embed_model: str):
        if not _HAS_ST:
            return
        self._embedder = SentenceTransformer(embed_model)
        texts = [c.text for c in self.chunks]
        self._dense_matrix = self._embedder.encode(texts, normalize_embeddings=True)

    def search(self, q: str, k: int = 8, alpha: float = 0.5) -> List[Chunk]:
        dense_scores = []
        if self._embedder is not None:
            qv = self._embedder.encode([q], normalize_embeddings=True)[0]
            sims = (self._dense_matrix @ qv)
            dense_scores = list(enumerate(sims))

        bm25_scores = list(enumerate(self._bm25.get_scores(q.split())))

        def normalize(lst):
            if not lst:
                return {}
            vals = np.array([score for _, score in lst])
            if vals.max() == vals.min():
                normed = np.ones_like(vals)
            else:
                normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
            return {i: float(v) for (i, _), v in zip(lst, normed)}

        nd = normalize(dense_scores)
        nb = normalize(bm25_scores)

        combined = {}
        for i in set(nd.keys()) | set(nb.keys()):
            combined[i] = alpha * nd.get(i, 0) + (1 - alpha) * nb.get(i, 0)

        top = sorted(combined.items(), key=lambda x: -x[1])[:k]
        return [self.chunks[i] for i, _ in top]
