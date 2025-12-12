"""
RAG pipeline:
- HybridRetriever (BM25 + embeddings)
- Guardrails: PII scrub, prompt injection, out-of-scope/harmful requests
- Strict RAG prompt with inline [Source: …] citations
- Optional local LoRA model for generation (TinyLlama + PEFT adapter)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from rag.retrieval import HybridRetriever
from rag.guardrails import scrub_pii, is_injection, is_out_of_scope_or_harmful
from libs.prompts import make_strict_rag_prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


# ---------------------------------------------------------------------
# Env + logging
# ---------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------

# NOTE: if LORA_PATH points to a PEFT adapter, we will read the base model
# from adapter_config.json automatically (safer than hardcoding BASE_MODEL).
LORA_PATH = os.getenv("LORA_PATH", "models/soc-assistant-lora/adapter")

# Optional override (only used if adapter config can’t be read for some reason)
BASE_MODEL_FALLBACK = os.getenv("LOCAL_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Use local model if set to 1, otherwise use OpenAI path.
USE_LOCAL_LORA = os.getenv("USE_LOCAL_LORA", "0") == "1"

# Conservative generation settings for tiny models
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
    """
    Loads base model + applies LoRA adapter at LORA_PATH.

    IMPORTANT FIXES (especially for Apple Silicon / MPS):
    - Avoid device_map="auto" (accelerate sharding can create meta tensors).
    - Set low_cpu_mem_usage=False to avoid meta tensors during load.
    - Move to device explicitly.
    - Read base model name from adapter_config.json via PeftConfig to ensure match.
    """
    global _local_model, _local_tokenizer

    if _local_model is not None and _local_tokenizer is not None:
        return _local_model, _local_tokenizer

    device = _best_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    # Prefer base model from PEFT adapter config (prevents mismatches)
    try:
        peft_cfg = PeftConfig.from_pretrained(LORA_PATH)
        base_model_name = peft_cfg.base_model_name_or_path
    except Exception as exc:
        logger.warning("Could not read PEFT config from %s (%s). Falling back to LOCAL_BASE_MODEL.", LORA_PATH, exc)
        base_model_name = BASE_MODEL_FALLBACK

    logger.info("Loading local base model: %s (device=%s, dtype=%s)", base_model_name, device, dtype)

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- CRITICAL: avoid meta tensors ---
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,  # <-- key for MPS/meta warnings
        device_map=None,          # <-- key to avoid accelerate meta/offload
    )

    # Move base to device explicitly
    base = base.to(device)

    logger.info("Applying LoRA adapter: %s", LORA_PATH)
    model = PeftModel.from_pretrained(base, LORA_PATH)

    model.eval()

    _local_model, _local_tokenizer = model, tok
    return _local_model, _local_tokenizer


def _extract_answer_from_full_text(full_text: str, prompt: str) -> str:
    """
    Heuristic cleanup:
    - If the model echoed the prompt, strip it.
    - Then, if there’s an ANSWER section marker, return what follows it.
    """
    text = full_text.strip()

    # Strip prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()

    # Your prompt uses these dividers:
    marker = "=======================\nANSWER"
    if marker in text:
        text = text.split(marker, 1)[-1].strip()

    return text.strip()


def _generate_local(prompt: str) -> str:
    model, tok = _load_local_lora_model()
    device = next(model.parameters()).device

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LOCAL_MAX_INPUT_TOKENS,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = LOCAL_TEMPERATURE > 0.0
    temperature = LOCAL_TEMPERATURE if do_sample else 1.0  # ignored when do_sample=False

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    full_text = tok.decode(out[0], skip_special_tokens=True)
    answer = _extract_answer_from_full_text(full_text, prompt)
    return answer or "I don't know."


# ---------------------------------------------------------------------
# Hybrid retriever cache (per-alpha)
# ---------------------------------------------------------------------

_retrievers: Dict[float, HybridRetriever] = {}


def _get_retriever(alpha: float = 0.5) -> HybridRetriever:
    global _retrievers
    if alpha not in _retrievers:
        logger.info("Building HybridRetriever(alpha=%.2f)", alpha)
        _retrievers[alpha] = HybridRetriever.from_processed_dir(alpha=alpha)
    return _retrievers[alpha]


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------

def _build_context(passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return ""

    passages_sorted = sorted(passages, key=lambda p: p.get("score", 0.0), reverse=True)
    top_score = passages_sorted[0].get("score", 0.0)

    if top_score > 0:
        filtered = [p for p in passages_sorted if p.get("score", 0.0) >= 0.6 * top_score]
    else:
        filtered = passages_sorted

    used = (filtered or passages_sorted[:5])[:8]

    max_chunk_chars = 900
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

def answer_query(q: str, top_k: int = 8, alpha: float = 0.7) -> Dict[str, Any]:
    original = q

    # 1) Out-of-scope / harmful
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

    # 4) Retrieval
    retriever = _get_retriever(alpha=alpha)
    passages = retriever.search(scrubbed, k=top_k)
    _log_retrieval_stats(scrubbed, passages, top_k)

    # 5) Prompt
    prompt = build_prompt(original, passages)

    # 6) Generate (local LoRA OR OpenAI)
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
            answer = (choice.content if choice else "") or "I don't know."

        guardrail_meta = {"blocked": False, "used_local_lora": USE_LOCAL_LORA}
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"Unable to generate an answer right now due to an error: {exc}"
        guardrail_meta = {"blocked": False, "error": exc.__class__.__name__}

    # 7) Citations
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