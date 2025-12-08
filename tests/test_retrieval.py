"""
Day 3 — Retrieval Tests

WHAT THESE TESTS DO
-------------------
We want to confirm:

1) Index files exist (BM25 + embeddings + chunks)
2) Retriever loads without errors
3) search() returns reasonable non-empty results
4) Output schema has the fields FastAPI/pipeline depends on

We skip tests automatically if indexes don't exist yet.
That way pytest doesn't scream before Day 3 build step.
"""

from pathlib import Path
import pytest
from dotenv import load_dotenv
load_dotenv()  # ensure .env values (OPENAI_API_KEY, etc.) are available
from rag.retrieval import (
    HybridRetriever,
    PROCESSED_DIR,
    BM25_PATH,
    EMB_PATH,
    CHUNKS_PATH
)


# This decorator tells pytest:
# "Skip this test if the index files don't exist yet."
@pytest.mark.skipif(
    not (BM25_PATH.exists() and EMB_PATH.exists() and CHUNKS_PATH.exists()),
    reason="Indexes not built yet. Run scripts/build_indexes.py first."
)
def test_retriever_loads_and_searches():
    """
    Sanity test:
      - load retriever
      - run a query
      - ensure we got hits and ordering by score works
    """
    r = HybridRetriever.from_processed_dir(PROCESSED_DIR)

    hits = r.search("MITRE ATT&CK", k=5)

    # must return something
    assert len(hits) > 0

    # chunks shouldn't be empty text
    assert all(h["text"].strip() for h in hits)

    # first score should be >= last score
    assert hits[0]["score"] >= hits[-1]["score"]


@pytest.mark.skipif(
    not (BM25_PATH.exists() and EMB_PATH.exists() and CHUNKS_PATH.exists()),
    reason="Indexes not built yet. Run scripts/build_indexes.py first."
)
def test_scores_have_expected_fields():
    """
    Contract test:
    Your pipeline + API expect these keys to exist.
    If one disappears, retrieval breaks downstream.
    """
    r = HybridRetriever.from_processed_dir(PROCESSED_DIR)

    hits = r.search("AI security risks", k=3)
    h = hits[0]

    # required fields for day 3/4/5 pipeline
    for key in ["id", "text", "score", "bm25_score", "semantic_score", "source"]:
        assert key in h