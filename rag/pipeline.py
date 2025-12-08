"""
RAG pipeline:
- HybridRetriever (BM25 + embeddings)
- Guardrails: PII scrub, prompt injection, out-of-scope/harmful requests
- Strict RAG prompt with inline [Source: …] citations
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from rag.retrieval import HybridRetriever
from rag.guardrails import scrub_pii, is_injection, is_out_of_scope_or_harmful
from libs.prompts import SYSTEM_PROMPT, make_strict_rag_prompt

# ---------------------------------------------------------------------
# Env + logging
# ---------------------------------------------------------------------

load_dotenv()  # ensure .env values (OPENAI_API_KEY, etc.) are available
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# OpenAI client (with dummy for tests / missing key)
# ---------------------------------------------------------------------

class _DummyCompletions:
    def create(self, *args, **kwargs):
        raise OpenAIError(
            "OPENAI_API_KEY is not set. Export it to call the OpenAI API."
        )


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class _DummyClient:
    def __init__(self):
        self.chat = _DummyChat()


def _get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("OPENAI_API_KEY not set; using dummy OpenAI client.")
        return _DummyClient()
    # FIX: explicitly pass api_key, but let the SDK handle env as usual
    return OpenAI(api_key=key)


client = _get_openai_client()


# ---------------------------------------------------------------------
# Hybrid retriever cache (per-alpha)
# ---------------------------------------------------------------------

# WEEK 7: support multiple retrievers keyed by alpha (BM25 vs embedding weight)
_retrievers: Dict[float, HybridRetriever] = {}


def _get_retriever(alpha: float = 0.5) -> HybridRetriever:
    """
    Lazily load a HybridRetriever for a given alpha.

    alpha controls the hybrid blend:
      - alpha close to 1.0 → more weight on BM25
      - alpha close to 0.0 → more weight on embeddings

    We cache per-alpha retrievers in a dict so each one only builds once.
    """
    global _retrievers
    if alpha not in _retrievers:
        logger.info("Building HybridRetriever(alpha=%.2f)", alpha)
        _retrievers[alpha] = HybridRetriever.from_processed_dir(alpha=alpha)
    return _retrievers[alpha]


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------

def _build_context(passages: List[Dict[str, Any]]) -> str:
    """
    Build a compact context string with explicit [Source: …] markers.

    Passages are dicts from HybridRetriever.search() like:
      {
        "id": "...",
        "text": "...",
        "score": ...,
        "source": "...",
        "page_num": ...,
        "metadata": {...}
      }
    """
    if not passages:
        return ""

    # Sort to guarantee descending hybrid score
    passages_sorted = sorted(passages, key=lambda p: p.get("score", 0.0), reverse=True)
    

    top_score = passages_sorted[0].get("score", 0.0)

    # FIX: handle case where top_score is 0.0 for all passages (avoid filtering out everything)
    if top_score > 0:
        filtered = [p for p in passages_sorted if p.get("score", 0.0) >= 0.6 * top_score]
    else:
        filtered = passages_sorted

    # Always cap to six passages to avoid overwhelming the model.
    # Fallback: if filtered is empty, at least use the top 3 overall.
    used = (filtered or passages_sorted[:5])[:8]

    max_chunk_chars = 900  # truncate long chunks
    context_lines: List[str] = []

    for p in used:
        meta = p.get("metadata") or {}

        # Try to construct a nice source tag:
        #   doc_id:chunk_id
        #   or doc_id:page
        #   or source:page
        doc_id = meta.get("doc_id") or p.get("source") or p.get("id") or "unknown"

        # FIX: check both metadata["page"] and p["page_num"]
        page_num = (
            meta.get("page_num")
            or meta.get("page")
            or p.get("page_num")
        )
        page_suffix = f":{page_num}" if page_num is not None else ""

        chunk_id = meta.get("chunk_id", "")

        if chunk_id:
            source_tag = f"{doc_id}:{chunk_id}"
        else:
            source_tag = f"{doc_id}{page_suffix}"

        text = (p.get("text") or "").strip()
        if not text:
            continue

        if len(text) > max_chunk_chars:
            text = text[: max_chunk_chars - 3].rstrip() + "..."

        context_lines.append(f"[Source: {source_tag}]\n{text}")

    return "\n\n".join(context_lines)


# ---------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------

def build_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    """
    Build a strict RAG prompt using the centralized prompt helper.

    Behavior:
    - Context is built via _build_context()
    - make_strict_rag_prompt() injects the strict RAG system rules:
      * only use context
      * must include citations
      * say 'I don't know.' if unsupported
    """
    context_text = _build_context(passages)
    # FIX: always pass the *original* question text (not scrubbed) here,
    # so the answer remains natural while retrieval uses scrubbed text.
    prompt = make_strict_rag_prompt(context=context_text, question=question)
    return prompt


# ---------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------

def _log_retrieval_stats(question: str, passages: List[Dict[str, Any]], top_k: int) -> None:
    """
    Log basic retrieval stats for observability.

    passages is a list[dict] from HybridRetriever.search().
    We log:
      - question (truncated)
      - requested top_k
      - actual number of passages
      - top, median, and min hybrid scores
    """
    if not passages:
        logger.info(
            "Retrieval stats: question=%r top_k=%d hits=0",
            question,
            top_k,
        )
        return

    scores = [p.get("score", 0.0) for p in passages]
    scores_sorted = sorted(scores, reverse=True)
    top_score = scores_sorted[0]
    min_score = scores_sorted[-1]
    median_score = scores_sorted[len(scores_sorted) // 2]

    logger.info(
        "Retrieval stats: question=%r top_k=%d hits=%d top=%.4f median=%.4f min=%.4f",
        question,
        top_k,
        len(passages),
        top_score,
        median_score,
        min_score,
    )


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------

def answer_query(q: str, top_k: int = 8, alpha: float = 0.7) -> Dict[str, Any]:
    """
    Main entry point for answering a question with RAG.

    Args:
        q: user question (raw)
        top_k: how many passages to retrieve
        alpha: hybrid weight between BM25 and semantic similarity.
               1.0 = BM25 only, 0.0 = semantic only, 0.5 = balanced.
    """
    original = q

    # 1) Out-of-scope / harmful requests get an immediate safe refusal.
    # FIX: run this on the *raw* question for maximum coverage.
    if is_out_of_scope_or_harmful(original):
        msg = (
            "I can't do that. The request is either unsafe or not supported by the "
            "provided documents, so I will not provide those details."
        )
        return {
            "question": original,
            "answer": msg,
            "citations": [],
            "guardrails": {"blocked": True, "reason": "out_of_scope_or_harmful"},
        }

    # 2) PII scrub on the remaining queries (used only for retrieval)
    scrubbed = scrub_pii(original)

    # 3) Prompt injection detection (run on scrubbed text to avoid leaking PII)
    if is_injection(scrubbed):
        return {
            "question": original,
            "answer": "Query blocked by guardrails.",
            "citations": [],
            "guardrails": {"blocked": True, "reason": "prompt_injection"},
        }

    # 4) Retrieval
    retriever = _get_retriever(alpha=alpha)
    passages = retriever.search(scrubbed, k=top_k)
    _log_retrieval_stats(scrubbed, passages, top_k)

    # 5) Build strict RAG prompt
    prompt = build_prompt(original, passages)

    # 6) Call LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        # FIX: guard against weird/empty responses
        choice = response.choices[0].message if response.choices else None
        answer = (choice.content if choice else "") or "I don't know."
        guardrail_meta = {"blocked": False}
    except OpenAIError as exc:
        logger.error("OpenAI chat completion failed: %s", exc)
        err_code = getattr(exc, "code", None) or exc.__class__.__name__
        detail = getattr(exc, "message", None) or str(exc)
        answer = (
            "Unable to generate an answer right now because the upstream model "
            f"returned an error ({err_code}): {detail}"
        )
        guardrail_meta = {"blocked": False, "llm_error": err_code}

    # 7) Build citations
    citations: List[Dict[str, Any]] = []
    for p in passages[:top_k]:
        citations.append(
            {
                "id": p.get("id"),
                "source": p.get("source"),
                "page_num": p.get("page_num"),
                "score": p.get("score"),
                "metadata": p.get("metadata", {}),
            }
        )

    return {
        "question": original,
        "answer": answer,
        "citations": citations,
        "guardrails": guardrail_meta,
    }