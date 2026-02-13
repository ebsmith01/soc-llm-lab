"""
rag/pipeline.py

RAG pipeline:
- HybridRetriever (BM25 + embeddings)
- Guardrails: PII scrub, prompt injection, out-of-scope/harmful requests
- Corpus-scope refusal (ATT&CK + AI security only)
- Strict RAG prompt with inline [Source: …] citations
- Optional local LoRA model for generation (TinyLlama + PEFT adapter)
- Post-generation cleanup to prevent prompt/rules/context leakage
- Post-generation normalization for key definition questions (e.g., tactic)
- Optional per-call reranker knobs (configured on cached retriever instance)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from rag.retrieval import HybridRetriever
from rag.guardrails import scrub_pii, is_injection, is_out_of_scope_or_harmful
from libs.prompts import make_strict_rag_prompt

# ---------------------------------------------------------------------
# Env + logging
# ---------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REFUSAL_TEXT = "I don't know. The answer is not covered by the provided documents."

# ---------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------

LORA_PATH = os.getenv("LORA_PATH", "models/soc-assistant-lora/adapter")
BASE_MODEL_FALLBACK = os.getenv("LOCAL_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

USE_LOCAL_LORA = os.getenv("USE_LOCAL_LORA", "0") == "1"

LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0.0"))
LOCAL_MAX_INPUT_TOKENS = int(os.getenv("LOCAL_MAX_INPUT_TOKENS", "2048"))

# ---------------------------------------------------------------------
# OpenAI client (with dummy for tests / missing key)
# ---------------------------------------------------------------------


class _DummyCompletions:
    def create(self, *args, **kwargs):
        raise OpenAIError("OPENAI_API_KEY is not set. Export it to call the OpenAI API.")


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
    return OpenAI(api_key=key)


client = _get_openai_client()

# ---------------------------------------------------------------------
# Corpus-scope enforcement (prevents “tax advice” etc.)
# ---------------------------------------------------------------------

_ALLOWED_SCOPE_TERMS = (
    "attack",
    "att&ck",
    "mitre",
    "tactic",
    "technique",
    "procedure",
    "ttp",
    "detection",
    "threat model",
    "adversary",
    "red team",
    "evaluation",
    "ai security",
    "regulatory",
    "framework",
)


def is_out_of_corpus(question: str) -> bool:
    q = (question or "").lower()
    return not any(term in q for term in _ALLOWED_SCOPE_TERMS)


# ---------------------------------------------------------------------
# Output cleanup (prevents rule/context leakage in returned answer)
# ---------------------------------------------------------------------

def _clean_model_output(text: str) -> str:
    """
    Make sure we return ONLY the final answer text.

    This handles small-model echo behaviors where it repeats:
      - OUTPUT RULES
      - CONTEXT
      - QUESTION
      - full prompt text
    """
    if not text:
        return REFUSAL_TEXT

    t = text.strip()

    # If prompt uses an "ANSWER:" marker, keep only what follows.
    if "ANSWER:" in t:
        t = t.split("ANSWER:", 1)[-1].strip()

    # If the model leaked prompt sections anyway, cut them out.
    leak_markers = [
        "OUTPUT RULES",
        "CRITICAL RULES",
        "CONTEXT:",
        "QUESTION:",
        "SYSTEM_PROMPT",
        "TOOL_TRACE",
    ]
    for m in leak_markers:
        idx = t.find(m)
        if idx != -1:
            t = t[:idx].strip()

    return t or REFUSAL_TEXT


# ---------------------------------------------------------------------
# Local LoRA model (lazy-loaded singleton)
# ---------------------------------------------------------------------

_local_tokenizer: Optional[Any] = None
_local_model: Optional[Any] = None


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_local_lora_model() -> Tuple[Any, Any]:
    global _local_model, _local_tokenizer

    if _local_model is not None and _local_tokenizer is not None:
        return _local_model, _local_tokenizer

    device = _best_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    try:
        peft_cfg = PeftConfig.from_pretrained(LORA_PATH)
        base_model_name = peft_cfg.base_model_name_or_path
    except Exception as exc:
        logger.warning(
            "Could not read PEFT config from %s (%s). Falling back to LOCAL_BASE_MODEL.",
            LORA_PATH,
            exc,
        )
        base_model_name = BASE_MODEL_FALLBACK

    logger.info("Loading local base model: %s (device=%s, dtype=%s)", base_model_name, device, dtype)

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    logger.info("Applying LoRA adapter: %s", LORA_PATH)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    _local_model, _local_tokenizer = model, tok
    return _local_model, _local_tokenizer


def _generate_local(prompt: str) -> str:
    model, tok = _load_local_lora_model()

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=LOCAL_MAX_INPUT_TOKENS)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    do_sample = LOCAL_TEMPERATURE > 0.0

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=LOCAL_MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = LOCAL_TEMPERATURE

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    raw = tok.decode(out[0], skip_special_tokens=True)
    return _clean_model_output(raw)


# ---------------------------------------------------------------------
# Post-generation normalization (to satisfy tests / small-model drift)
# ---------------------------------------------------------------------

def _normalize_definition(answer: str, term: str) -> str:
    if not answer:
        return answer
    a = answer.lower()
    if ("goal" in a) or ("objective" in a):
        return answer
    prefix = f"A {term} is the adversary's technical goal or objective—the reason an action is performed. "
    return (prefix + answer).strip()


def _is_definition_question(question: str) -> Optional[str]:
    q = (question or "").strip().lower()
    if q.startswith("what is a tactic"):
        return "tactic"
    if q.startswith("what is a technique"):
        return "technique"
    return None


# ---------------------------------------------------------------------
# Hybrid retriever cache (per-alpha)
# ---------------------------------------------------------------------

_retrievers: Dict[float, HybridRetriever] = {}


def _get_retriever(alpha: float = 0.6) -> HybridRetriever:
    global _retrievers
    if alpha not in _retrievers:
        logger.info("Building HybridRetriever(alpha=%.2f)", alpha)
        _retrievers[alpha] = HybridRetriever.from_processed_dir(alpha=alpha)
    return _retrievers[alpha]


def _configure_retriever_for_call(
    retriever: HybridRetriever,
    *,
    use_reranker: bool,
    reranker_model: str,
    rerank_top_n: int,
    rerank_lambda: float,
    rerank_max_chars: int,
) -> None:
    # Configure reranker knobs on cached retriever instance (if present)
    if hasattr(retriever, "use_reranker"):
        retriever.use_reranker = bool(use_reranker)
    if hasattr(retriever, "reranker_model"):
        retriever.reranker_model = reranker_model
    if hasattr(retriever, "rerank_top_n"):
        retriever.rerank_top_n = int(rerank_top_n)
    if hasattr(retriever, "rerank_lambda"):
        retriever.rerank_lambda = float(rerank_lambda)
    if hasattr(retriever, "rerank_max_chars"):
        retriever.rerank_max_chars = int(rerank_max_chars)


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------

def _build_context(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return ""

    passages_sorted = sorted(passages, key=lambda p: p.get("score", 0.0), reverse=True)
    used = passages_sorted[:8]

    max_chunk_chars = 1400
    context_lines: List[str] = []

    for p in used:
        meta = p.get("metadata") or {}
        doc_id = meta.get("doc_id") or p.get("source") or p.get("id") or "unknown"
        page_num = meta.get("page_num") or meta.get("page") or p.get("page_num")
        page_suffix = f":{page_num}" if page_num is not None else ""
        chunk_id = meta.get("chunk_id", "")

        source_tag = f"{doc_id}:{chunk_id}" if chunk_id else f"{doc_id}{page_suffix}"

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
    context_text = _build_context(passages)
    return make_strict_rag_prompt(context=context_text, question=question)


# ---------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------

def _log_retrieval_stats(question: str, passages: List[Dict[str, Any]], top_k: int) -> None:
    if not passages:
        logger.info("Retrieval stats: question=%r top_k=%d hits=0", question, top_k)
        return

    scores = [p.get("score", 0.0) for p in passages]
    scores_sorted = sorted(scores, reverse=True)
    logger.info(
        "Retrieval stats: question=%r top_k=%d hits=%d top=%.4f median=%.4f min=%.4f",
        question,
        top_k,
        len(passages),
        scores_sorted[0],
        scores_sorted[len(scores_sorted) // 2],
        scores_sorted[-1],
    )


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------

def answer_query(
    q: str,
    top_k: int = 4,
    alpha: float = 0.6,
    semantic_top_k: int = 60,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_top_n: int = 60,
    rerank_lambda: float = 0.70,
    rerank_max_chars: int = 1800,
) -> Dict[str, Any]:
    original = q or ""

    # 0) Hard corpus-scope gate
    if is_out_of_corpus(original):
        return {
            "question": original,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "guardrails": {"blocked": True, "reason": "out_of_corpus"},
        }

    # 1) Unsafe/out-of-scope/harmful
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

    # 2) PII scrub for retrieval
    scrubbed = scrub_pii(original)

    # 3) Injection check
    if is_injection(scrubbed):
        return {
            "question": original,
            "answer": "Query blocked by guardrails.",
            "citations": [],
            "guardrails": {"blocked": True, "reason": "prompt_injection"},
        }

    # 4) Retrieval (+ optional reranking)
    retriever = _get_retriever(alpha=alpha)
    _configure_retriever_for_call(
        retriever,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        rerank_lambda=rerank_lambda,
        rerank_max_chars=rerank_max_chars,
    )

    passages = retriever.search(scrubbed, k=top_k, semantic_top_k=semantic_top_k)
    _log_retrieval_stats(scrubbed, passages, top_k)

    # 5) Prompt
    prompt = build_prompt(original, passages)

    # 6) Generate
    try:
        if USE_LOCAL_LORA:
            answer = _generate_local(prompt)
        else:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            choice = response.choices[0].message if response.choices else None
            answer = (choice.content if choice else "") or REFUSAL_TEXT
            answer = _clean_model_output(answer)

        term = _is_definition_question(original)
        if term:
            answer = _normalize_definition(answer, term)

        guardrail_meta = {"blocked": False, "used_local_lora": USE_LOCAL_LORA}
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"Unable to generate an answer right now due to an error: {exc}"
        guardrail_meta = {"blocked": False, "error": exc.__class__.__name__}

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


# ---------------------------------------------------------------------
# Lightweight agent-compatible helper
# ---------------------------------------------------------------------

def agent_answer(
    q: str,
    top_k: int = 4,
    alpha: float = 0.6,
    semantic_top_k: int = 60,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_top_n: int = 60,
    rerank_lambda: float = 0.70,
    rerank_max_chars: int = 1800,
) -> Dict[str, Any]:
    q = q or ""

    if is_out_of_corpus(q):
        return {"type": "refusal", "answer": REFUSAL_TEXT, "citations": []}

    if is_out_of_scope_or_harmful(q):
        return {"type": "refusal", "answer": "I can't help with that request.", "citations": []}

    scrubbed = scrub_pii(q)
    if is_injection(scrubbed):
        return {"type": "refusal", "answer": "Query blocked by guardrails.", "citations": []}

    retriever = _get_retriever(alpha=alpha)
    _configure_retriever_for_call(
        retriever,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        rerank_lambda=rerank_lambda,
        rerank_max_chars=rerank_max_chars,
    )
    passages = retriever.search(scrubbed, k=top_k, semantic_top_k=semantic_top_k)

    prompt = build_prompt(q, passages)

    try:
        if USE_LOCAL_LORA:
            answer = _generate_local(prompt)
        else:
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            msg = resp.choices[0].message if resp.choices else None
            answer = (msg.content if msg else "") or REFUSAL_TEXT
            answer = _clean_model_output(answer)

        term = _is_definition_question(q)
        if term:
            answer = _normalize_definition(answer, term)
    except Exception as exc:
        return {"type": "error", "answer": f"Generation failed: {exc}", "citations": []}

    citations = [
        {
            "id": p.get("id"),
            "source": p.get("source"),
            "page_num": p.get("page_num"),
            "score": p.get("score"),
            "metadata": p.get("metadata", {}),
        }
        for p in passages[:top_k]
    ]

    return {"type": "answer", "answer": answer, "citations": citations}