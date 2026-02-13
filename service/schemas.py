"""
Pydantic schemas for request/response payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to route through the RAG pipeline.")
    use_local_lora: Optional[bool] = Field(None, description="Override to force local LoRA generation.")
    top_k: int = Field(4, ge=1, le=20, description="Number of passages to retrieve.")
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Hybrid retriever weighting factor.")


class Citation(BaseModel):
    id: Optional[str] = Field(None, description="Document chunk id.")
    source: Optional[str] = Field(None, description="Document source identifier.")
    page_num: Optional[int] = Field(None, description="Page number if available.")
    score: Optional[float] = Field(None, description="Retriever score.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GuardrailMeta(BaseModel):
    blocked: bool = False
    reason: Optional[str] = None
    used_local_lora: Optional[bool] = None
    error: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    guardrails: GuardrailMeta = Field(default_factory=GuardrailMeta)
