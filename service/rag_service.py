"""
Thin wrapper around the core RAG pipeline so the API layer stays lean.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rag import pipeline


class RagService:
    def __init__(self) -> None:
        # In a fuller build, you could inject retriever instances, telemetry, etc.
        pass

    def answer(self, question: str, *, use_local_lora: Optional[bool] = None, top_k: int = 6, alpha: float = 0.7) -> Dict[str, Any]:
        if use_local_lora is not None:
            pipeline.USE_LOCAL_LORA = bool(use_local_lora)

        result = pipeline.answer_query(question, top_k=top_k, alpha=alpha)

        # Normalize to schema fields expected by the API response.
        return {
            "question": result.get("question", question),
            "answer": result.get("answer", "I don't know."),
            "citations": result.get("citations", []),
            "guardrails": result.get("guardrails", {}),
        }

